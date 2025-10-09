import sys
import argparse
import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoTokenizer, AutoModelForCausalLM

# minimal files for a causal LM; shrinks download time a lot
ALLOW_PATTERNS = [
    # configs
    "config.json", "generation_config.json",
    # tokenizer (fast or slow)
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt",          # BPE (GPT/OPT)
    "tokenizer.model",                    # sentencepiece (Llama/Qwen)
    # weights
    "model.safetensors", "*.safetensors",
    "pytorch_model.bin", "pytorch_model-*.bin",
]
# working on your env (fast OK)
""" "gpt-neo-125M": "EleutherAI/gpt-neo-125M",
    "opt-125m":     "facebook/opt-125m",
    "llama3.2-1b":  "meta-llama/Llama-3.2-1B",              # gated
    "qwen-0.5b":    "Qwen/Qwen2.5-0.5B"," """
"""DEFAULT_MODELS = {

    #"deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # (skip gpt2/distilgpt2 on your Windows env for now)
}"""



DEFAULT_MODELS = {
    "gpt-neo-125m":           "EleutherAI/gpt-neo-125M",                 # tiny, always works
    "tinyllama-1.1b-chat":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",      # light chat model
    "qwen2.5-1.5b-instruct":  "Qwen/Qwen2.5-1.5B-Instruct",              # strong small model
    "phi-2-2.7b":             "microsoft/phi-2",                         # borderline on 8 GB
}

def verify_local_model(local_dir: Path) -> bool:
    """Try to load tokenizer+model from local folder (no internet)."""
    try:
        tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True, use_fast=True)
    except Exception:
        # fall back to slow (sentencepiece)
        tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(str(local_dir), local_files_only=True)
    # light check to ensure basic attributes exist
    _ = tok.eos_token_id
    _ = mdl.config.model_type
    return True

def prefetch_one(repo_id: str, root: Path, token: Optional[str]) -> Path:
    local = root / repo_id.replace("/", "__")
    local.mkdir(parents=True, exist_ok=True)
    marker = local / ".complete"

    if marker.exists():
        print(f"[SKIP] {repo_id} already marked complete at {local}")
        return local

    print(f"[DL] {repo_id} → {local}")
    try:
        p = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local),
            local_dir_use_symlinks=False,  # Windows friendly
            allow_patterns=ALLOW_PATTERNS,
            token=token,
        )
        print(f"[OK]  snapshot at {p}")
    except HfHubHTTPError as e:
        print(f"[ERR] {repo_id}: {e}")
        if "401" in str(e) or "gated" in str(e).lower():
            print("      → This repo is gated/private. Make sure you have access and are logged in.")
        raise

    # verify local load to catch partial/failed downloads early
    try:
        verify_local_model(local)
    except Exception as e:
        print(f"[WARN] verification failed for {repo_id}: {e}")
        print("       (Will still keep files; you can re-run later.)")
    else:
        marker.touch()

    return local

def main():
    ap = argparse.ArgumentParser("Prefetch HF models into ./models/<org__name>")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS.values()),
                    help="List of HF repo IDs to download. Default: a known-good set.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (optional). If omitted, uses env HF_TOKEN or CLI login.")
    ap.add_argument("--out_dir", default="models", help="Local snapshot root directory.")
    args = ap.parse_args()

    # login if token provided
    if args.token:
        try:
            login(token=args.token)
        except Exception as e:
            print(f"[WARN] login failed with provided token: {e} (continuing)")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Will download these repos:")
    for m in args.models:
        print("  -", m)

    for repo in args.models:
        try:
            prefetch_one(repo, out_root, token=args.token)
        except Exception:
            # do not stop the whole batch; continue with others
            continue

    print("\n[DONE] Prefetch complete.")
    print("You can now load from local paths, e.g.:")
    print("  local_dir = 'models/' + repo_id.replace('/', '__')")
    print("  AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True/False)")
    print("  AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)")

if __name__ == "__main__":
    main()
