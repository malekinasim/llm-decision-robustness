# scripts/extract_features.py
import sys, os, argparse
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.util import load_model_and_tokenizer, split_train_test, get_device
from src.io import load_prompts_with_options, ensure_dirs
from src.hooks import mcq_alllayer_hiddens, single_alllayer_hiddens
from src.probing import build_pairs_single  # برای ساخت جفت‌های single (gold/decoy)
from src.feature_cache import save_feature_cache

def _build_MCQ_features(items, model, tokenizer):
    """Return X_layers, y, qids  (per-option instances)."""
    X_layers = None; y = []; qids = []
    for it in items:
        q, options, gold = it["question"], it["options"], it["answer"]
        # نکته: ensure last non-PAD == last token of option
        tokenizer.padding_side = "right"
        layer_dicts = mcq_alllayer_hiddens(model, tokenizer, q, options,
                                           skip_embedding=True, add_space_between=True)
        L = len(layer_dicts)
        if X_layers is None:
            X_layers = {li: [] for li in range(L)}
        for opt in options:
            y.append(1 if opt == gold else 0)
            qids.append(it.get("id", q))  # شناسه سؤال
            for li in range(L):
                X_layers[li].append(layer_dicts[li][opt].cpu().numpy())
    X_layers = {li: np.vstack(m) for li, m in X_layers.items()}
    return X_layers, np.asarray(y, np.int64), np.asarray(qids)

def _build_SINGLE_features(pairs, model, tokenizer, batch=32):
    """pairs: list of (prompt, completion, label, group_id) from build_pairs_single"""
    X_layers = None; y = []; qids = []
    for i in range(0, len(pairs), batch):
        pk = pairs[i:i+batch]
        prompts = [str(p[0]) for p in pk]
        comps   = [str(p[1]) for p in pk]
        labs    = [int(p[2]) for p in pk]
        gids    = [p[3] if len(p) > 3 else f"g{i+j}" for j, p in enumerate(pk)]

        tokenizer.padding_side = "right"
        layer_vecs = single_alllayer_hiddens(model, tokenizer, prompts, comps,
                                             skip_embedding=True, add_space_between=True)  # List[L] of [b,D]

        L = len(layer_vecs)
        if X_layers is None:
            X_layers = {li: [] for li in range(L)}
        for li in range(L):
            X_layers[li].append(layer_vecs[li].cpu().numpy())
        y.extend(labs); qids.extend(gids)

    X_layers = {li: np.vstack(m) for li, m in X_layers.items()}
    return X_layers, np.asarray(y, np.int64), np.asarray(qids)

def main():
    ap = argparse.ArgumentParser("Phase-1: extract & cache per-layer features (no viz, no probes)")
    ap.add_argument("--task", choices=["mcq", "single"], required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--dataset", default=str(REPO_ROOT / "data" / "prompt_pool.json"))
    ap.add_argument("--test_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--remote", default=False)
    ap.add_argument("--compute_dtype", choices=["auto","float32","float16","bf16"], default="auto",
                help="dtype for model forward (affects hidden states)")
    ap.add_argument("--store_dtype", choices=["float32","float16"], default="float32",
                help="dtype on disk for cached features")

    args = ap.parse_args()

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device,compute_dtype=args.compute_dtype, remote=args.remote)
    model_path = args.model.replace("/", "__")
    cache_dir  = os.path.join(args.out_root, "features", model_path, args.task) 
    os.makedirs(cache_dir, exist_ok=True)

    mcq_items, single_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=False)

    if args.task == "mcq":
        items = [it for it in mcq_items if it.get("answer") in it.get("options", []) and len(it.get("options", [])) >= 2]
        train, test = split_train_test(items, test_ratio=args.test_ratio, seed=args.seed)
        Xtr_layers, ytr, qtr = _build_MCQ_features(train, model, tokenizer)
        Xte_layers, yte, qte = _build_MCQ_features(test,  model, tokenizer)
        save_feature_cache(os.path.join(cache_dir, "train.npz"), Xtr_layers, ytr, qtr)
        save_feature_cache(os.path.join(cache_dir, "test.npz"),  Xte_layers, yte, qte)
        print("[MCQ] cached:", cache_dir, {li: Xtr_layers[li].shape for li in sorted(Xtr_layers)})

    else:
        pairs = build_pairs_single(model=model, tok=tokenizer, device=device,
                                   items=single_items, negatives_per=2)
        train_pairs, test_pairs = split_train_test(pairs, test_ratio=args.test_ratio, seed=args.seed)
        Xtr_layers, ytr, qtr = _build_SINGLE_features(train_pairs, model, tokenizer)
        Xte_layers, yte, qte = _build_SINGLE_features(test_pairs,  model, tokenizer)
        save_feature_cache(os.path.join(cache_dir, "train.npz"), Xtr_layers, ytr, qtr)
        save_feature_cache(os.path.join(cache_dir, "test.npz"),  Xte_layers, yte, qte)
        print("[SINGLE] cached:", cache_dir, {li: Xtr_layers[li].shape for li in sorted(Xtr_layers)})

if __name__ == "__main__":
    main()
