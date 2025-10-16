
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
    # همان لاجیک خودت؛ اینجا نمونهٔ ساده:
    root = Path("models")
    return root / model_name.replace("/", "__")

def _to_bool(x):
    if isinstance(x, bool): return x
    if x is None: return False
    return str(x).lower() in {"1","true","yes","y"}

def _has_local_snapshot(ldir: Path) -> bool:
    # robust: safetensors یا bin، با یا بدون index
    has_cfg = (ldir / "config.json").exists()
    has_st  = any(ldir.glob("*.safetensors"))
    has_bin = any(ldir.glob("pytorch_model*.bin"))
    return has_cfg and (has_st or has_bin)

def load_model_and_tokenizer(
    model_name: str,
    device=None,
    compute_dtype: str = "auto",   # "auto" | "float32" | "float16" | "bf16"
    remote: bool = False,
    trust_remote_code: bool = True
):
    """
    remote=True  -> همیشه از HF
    remote=False -> اگر اسنپ‌شات لوکال بود همان، وگرنه HF
    compute_dtype کنترل دقت محاسبه (FP32/FP16/BF16) را مشخص می‌کند.
    """
    # پایداری بیشتر روی ویندوز
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    remote = _to_bool(remote)
    local_dir = _local_dir_for(model_name)
    use_local = _has_local_snapshot(local_dir)

    src = model_name if (remote or not use_local) else str(local_dir)
    from_local = (src == str(local_dir))

    dtype_map = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
    }
    req_dtype = dtype_map.get(str(compute_dtype).lower(), None)

    # اگر GPU دارید و BF16 خواستید اما کارت پشتیبانی نمی‌کند، به FP16 برگرد
    has_cuda = torch.cuda.is_available()
    if req_dtype is torch.bfloat16 and not (has_cuda and torch.cuda.is_bf16_supported()):
        print("[warn] bf16 not supported on this GPU; falling back to float16")
        req_dtype = torch.float16

    # tokenizer
    print(f"[load] tokenizer from {'local' if from_local else 'HF'}...")
    tok = AutoTokenizer.from_pretrained(
        src,
        local_files_only=from_local,
        use_fast=True,
        trust_remote_code=trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    tok.padding_side = "right"

    # model
    print(f"[load] model from {'local' if from_local else 'HF'}...")
    model = AutoModelForCausalLM.from_pretrained(
        src,
        local_files_only=from_local,
        trust_remote_code=trust_remote_code,
        torch_dtype=req_dtype,                 
        device_map=("auto" if has_cuda else None),
        low_cpu_mem_usage=True,
        attn_implementation="eager",         
    )

    # اگر device صراحتاً داده شده، آن را اعمال کن
    if device is not None:
        model.to(device)

    model.eval()
    eff_dtype = getattr(model, "dtype", None)
    print(f"[ready] {model.__class__.__name__} | torch_dtype={eff_dtype} | src={'local' if from_local else 'HF'}")
    return model, tok

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