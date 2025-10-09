
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from pathlib import Path

# Config & device
# -----------------------------
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def _local_dir_for(model_name: str) -> Path:
    """Map a repo id to local snapshot path."""
    return Path("models") / model_name.replace("/", "__")

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)

def load_model_and_tokenizer(model_name: str, device=None, torch_dtype=None, remote=False):
    """
    remote=True  -> حتماً از اینترنت (HF) بگیر
    remote=False -> اگر فولدر محلی موجود بود از لوکال، وگرنه از HF
    """
    # پایداری بیشتر روی ویندوز
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    remote = _to_bool(remote)
    local_dir = _local_dir_for(model_name)
    use_local = (local_dir / "config.json").exists() and (local_dir / "model.safetensors").exists()

    # منبع لود
    src = model_name if (remote or not use_local) else str(local_dir)
    from_local = (src == str(local_dir))

    # انتخاب dtype امن: CPU=fp32 ، GPU=bfloat16 (یا fp16 اگر لازم بود)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # tokenizer
    print(f"[load] tokenizer from {'local' if from_local else 'HF'}...")
    tok = AutoTokenizer.from_pretrained(
        src,
        local_files_only=from_local,    # اگر از لوکال می‌خوانیم، آنلاین نشود
        use_fast=True,
        trust_remote_code=True          # برای خانواده Qwen/DeepSeek لازم است
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model
    print(f"[load] model from {'local' if from_local else 'HF'}...")
    has_cuda = torch.cuda.is_available()
    # توجه: از `dtype=` به‌جای `torch_dtype=` استفاده می‌کنیم (هشدار رفع می‌شود)
    mdl = AutoModelForCausalLM.from_pretrained(
        src,
        local_files_only=from_local,
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map=("auto" if has_cuda else None),  # GPU اگر هست استفاده شود
        low_cpu_mem_usage=True,                      # RAM کمتر هنگام لود
        attn_implementation="eager",                 # پایدارتر روی ویندوز/عمومی
    )

    # انتقال دستی فقط وقتی device صراحتاً داده شده
    if device is not None and device.type == "cpu" and has_cuda:
        # اگر کاربر خواسته CPU باشد ولی device_map=auto رفته GPU، برگردان روی CPU
        mdl.to(device)

    mdl.eval()
    print(f"[ready] {mdl.__class__.__name__} | dtype={mdl.dtype} | src={'local' if from_local else 'HF'}")
    return mdl, tok


def split_train_test(items, test_ratio=0.3, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_test = max(1, int(len(items) * test_ratio))
    test_idx = set(idx[:n_test])
    train, test = [], []
    for i, it in enumerate(items):
        (test if i in test_idx else train).append(it)
    return train, test