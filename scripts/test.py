import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_info()  # لاگ بیشتر ببینیم

local_dir = r"models\deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B"

# جلوگیری از تلاش برای GPU/FlashAttention
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU را بی‌اثر کن
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"    # بی‌اثر (برای اطمینان)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # اگر مک بود، بی‌خطره

print("Step 1: load tokenizer...")
tok = AutoTokenizer.from_pretrained(
    local_dir,
    local_files_only=True,
    use_fast=True,
    trust_remote_code=True
)
print("OK tokenizer, vocab:", len(tok))

print("Step 2: load model (CPU, eager attention)...")
t0 = time.time()
mdl = AutoModelForCausalLM.from_pretrained(
    local_dir,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float32,        # روی CPU حتما float32
    device_map=None,                  # صراحتاً CPU
    low_cpu_mem_usage=False,          # برای برخی مدل‌ها امن‌تره
    attn_implementation="eager"       # از SDPA/flash صرف‌نظر کن
)
print("OK model loaded in", round(time.time()-t0, 1), "sec")
mdl.eval()

# یک forward خیلی کوچک برای اطمینان
print("Step 3: tiny forward...")
inp = tok("hello", return_tensors="pt")
with torch.no_grad():
    out = mdl(**inp)
print("Done. last_hidden_state shape:", tuple(out.last_hidden_state.shape))
