# scripts/run_massmean.py
import sys
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.probing import build_pairs_single,eval_binary_from_pairs
from src.util import load_model_and_tokenizer, split_train_test, get_device  
from src.io import load_prompts_with_options, ensure_dirs                    
from src.hooks import mcq_alllayer_hiddens, single_alllayer_hiddens         
from src.mass_mean import (                                                
    mass_mean_fit_per_layer, mass_mean_eval_per_layer, mass_mean_fit_from_pairs
)
from src.viz import (                                                      
   plot_combined_diagnostics, plot_layer_acc_curve, plot_sep_pca_lda
)
from src.linear_probes import fit_eval_probes_per_layer , scores_for_probe ,tune_probes_on_layer                   


def _layer_metric_from_all_probes(probe_results: dict, mode: str = "max") -> dict[int, float]:
    """
    Aggregate AUROC across methods per layer.
    mode: 'max' (best probe per layer) or 'mean'
    returns {layer: aggregated_auroc}
    """
    methods = [m for m in probe_results.keys() if probe_results.get(m)]
    layers = sorted({li for m in methods for li in probe_results[m].keys()})
    agg = {}
    for li in layers:
        vals = []
        for m in methods:
            r = probe_results[m].get(li, {})
            au = r.get('auroc', float('nan'))
            if np.isfinite(au):
                vals.append(au)
        agg[li] = (max(vals) if mode == "max" else (sum(vals)/len(vals))) if vals else float('nan')
    return agg

def _best_layer_per_method(probe_results: dict) -> dict[str, int]:
    """Return the best layer index (by AUROC) for each probe method."""
    bests = {}
    for method, res in probe_results.items():
        if not res:
            continue  # skip if no results for this method
        # Find the layer with maximum AUROC for this method
        best_layer = max(res.items(), key=lambda kv: kv[1].get('auroc', -1))[0]
        bests[method] = best_layer
    return bests


def _spec_layers_without_best(acc_map_for_pick: dict, spec: str, L_total: int) -> list[int]:
    """
    Parse spec (e.g., "best,first,mid,last,10") but IGNORE 'best'.
    Return sorted unique layers for tokens other than 'best'.
    acc_map_for_pick: {layer: metric} only to know layer index range (optional here).
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    layers = []
    all_layers = sorted(acc_map_for_pick.keys()) if acc_map_for_pick else list(range(L_total))

    if "first" in tokens and all_layers:
        layers.append(all_layers[0])
    if "mid" in tokens and all_layers:
        layers.append(all_layers[len(all_layers)//2])
    if "last" in tokens and all_layers:
        layers.append(all_layers[-1])

    # numeric layers in spec
    for t in tokens:
        if t.isdigit():
            li = int(t)
            if 0 <= li < L_total:
                layers.append(li)

    # unique + sorted
    out = []
    seen = set()
    for li in layers:
        if li not in seen:
            out.append(li); seen.add(li)
    return sorted(out)


def _build_MCQ_prompt_feature(dataset,model,tokenizer):
    X_layers = None; y = []
    for it in dataset:
        q, options, gold = it["question"], it["options"], it["answer"]
        layer_dicts = mcq_alllayer_hiddens(model, tokenizer, q, options, skip_embedding=True, add_space_between=True)  #:contentReference[oaicite:8]{index=8}
        L = len(layer_dicts)
        if X_layers is None:
            X_layers = {li: [] for li in range(L)}
        for opt in options:
            lab = 1 if opt == gold else 0
            y.append(lab)
            for li in range(L):
                X_layers[li].append(layer_dicts[li][opt].numpy())
    X_layers = {li: np.vstack(m) for li, m in X_layers.items()}
    y = np.array(y, dtype=np.int64)
    return X_layers,y

def _build_single_propmpt_feature(dataset, model, tokenizer):
    """Build per-layer feature matrices (X_layers) and labels y for SINGLE task.
    Accepts either:
      - pairs: list of tuples (prompt, completion, label, group_id), OR
      - old dict items: [{"question":..., "answer":..., "negs":[...]/"neg":...}, ...]
    Returns:
      X_layers: dict[layer_idx] -> np.ndarray [N, D]
      y: np.ndarray [N] of int (0/1)
    """
    BATCH = 32
    X_layers = None
    y = []

    # Detect input format
    is_pair = False
    if len(dataset) > 0:
        is_pair = isinstance(dataset[0], (tuple, list)) and len(dataset[0]) >= 3

    v_prompts, v_comps, v_labels = [], [], []

    if is_pair:
        # New format: list of (prompt, completion, label, group_id)
        for p in dataset:
            pr, comp, lab = str(p[0]), str(p[1]), int(p[2])
            v_prompts.append(pr); v_comps.append(comp); v_labels.append(lab)
    else:
        # Old dict format: build (+) gold and one (-) neg per item
        for it in dataset:
            task = (it.get("task") or "").lower()
            if task not in {"single", "singel"}:
                continue
            pr  = str(it["question"])
            pos = str(it["answer"])
            v_prompts.append(pr); v_comps.append(pos);   v_labels.append(1)
            neg = it["negs"][0] if it.get("negs") else it.get("neg", "")
            if neg is not None and neg != "":
                v_prompts.append(pr); v_comps.append(str(neg)); v_labels.append(0)

    # Batch forward
    for i in range(0, len(v_prompts), BATCH):
        prompts = [str(p) for p in v_prompts[i:i+BATCH]]
        comps   = [str(c) for c in v_comps[i:i+BATCH]]
        labs    = [int(l) for l in v_labels[i:i+BATCH]]

        # IMPORTANT: don't prepend " " here; rely on add_space_between=True
        layer_vecs = single_alllayer_hiddens(
            model, tokenizer, prompts, comps,
            skip_embedding=True, add_space_between=True
        )  # List[L] of tensors [b, D]

        L = len(layer_vecs)
        if X_layers is None:
            X_layers = {li: [] for li in range(L)}
        for li in range(L):
            V = layer_vecs[li].numpy()  # [b, D]
            X_layers[li].append(V)
        y.extend(labs)

    # Stack blocks
    X_layers = {li: np.vstack(m) for li, m in X_layers.items()}
    y = np.array(y, dtype=np.int64)
    return X_layers, y



def main():
    print("start")
    parser = argparse.ArgumentParser(description="Linear probes (Mass-Mean, LDA, LogReg, LinSVM) per layer for MCQ/Single.")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "prompt_pool.json"))
    parser.add_argument("--task", choices=["mcq", "single"], required=True)
    parser.add_argument("--method", choices=["massmean","lda","logreg","linsvm"], required=True,default="massmean")
    parser.add_argument("--mcq_test_ratio", type=float, default=0.3)
    parser.add_argument("--single_negatives_per", type=int, default=2)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", default=str(REPO_ROOT))
    parser.add_argument("--viz_layers", default="best,first,mid,last",
                        help="comma list: best,first,mid,last or explicit layer indices (e.g., '3,10')")
    parser.add_argument("--remote", default=False)
    
    args = parser.parse_args()
    print(args.remote)
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.model, device, remote=args.remote)
    mcq_items, single_items = load_prompts_with_options(args.dataset, tokenizer, require_single_token=False)  # enforce single-token
    model_path = args.model.replace("/", "__")
    tables_dir = os.path.join(args.out_root, "reports", model_path, "tables",args.task)
    figs_dir   = os.path.join(args.out_root, "reports", model_path, "figures",args.task)
    ensure_dirs(tables_dir, figs_dir)
    
    method =args.method
    # ---------------------------- MCQ ----------------------------
    if args.task == "mcq":
        mcq = [it for it in mcq_items if it.get("answer") in it.get("options", []) and len(it.get("options", [])) >= 2]
        if not mcq:
            print("[MCQ] No valid MCQ items found."); return

        # ✦ اول split، بعد هر کاری
        train, test = split_train_test(mcq, test_ratio=args.mcq_test_ratio, seed=args.seed)

        # ساخت ویژگی‌ها
        Xtr_layers, ytr = _build_MCQ_prompt_feature(train, model, tokenizer)
        Xte_layers, yte = _build_MCQ_prompt_feature(test,  model, tokenizer)

        probe_results = {}
        best_layer = None

        if method == 'massmean':
            # baseline: fit + eval
            W = mass_mean_fit_per_layer(train, model, tokenizer, mcq_alllayer_hiddens)
            per_layer_acc, best_layer, best_acc = mass_mean_eval_per_layer(test, W, model, tokenizer, mcq_alllayer_hiddens)
            out_csv = os.path.join(tables_dir, "massmean_mcq_perlayer_acc.csv")
            pd.DataFrame([{"layer": li, "acc": acc, "n_test": len(test)} for li, acc in per_layer_acc.items()]).to_csv(out_csv, index=False)
            plot_layer_acc_curve(per_layer_acc, f"MCQ Mass-Mean per-layer ({args.model})", os.path.join(figs_dir, "massmean_mcq_perlayer_acc.png"))
            print(f"[SAVE] {out_csv}")
        else:
           
            keys = sorted(Xtr_layers.keys())
            mid_li = keys[len(keys)//2]
            best_params = tune_probes_on_layer(Xtr_layers[mid_li], ytr, random_state=args.seed, method=method)
            print("[TUNE] best params:", best_params)
            # ✦ املای آرگومان method را درست کن
            probe_results = fit_eval_probes_per_layer(Xtr_layers, ytr, Xte_layers, yte, method=method, best_params=best_params)

            # ذخیره CSV برای همان متد
            for m, res in probe_results.items():
                rows = []
                for li, r in sorted(res.items()):
                    rows.append({
                        "layer": li, "acc": r.get("acc", np.nan), "auroc": r.get("auroc", np.nan),
                        "fisher": r.get("fisher", np.nan), "thr0": r.get("thr0", 0.0), "thr_star": r.get("thr_star", 0.0),
                        "tp0": r.get("cm_thr0", {}).get("tp", np.nan), "fp0": r.get("cm_thr0", {}).get("fp", np.nan),
                        "tn0": r.get("cm_thr0", {}).get("tn", np.nan), "fn0": r.get("cm_thr0", {}).get("fn", np.nan),
                        "tp*": r.get("cm_thr_star", {}).get("tp", np.nan), "fp*": r.get("cm_thr_star", {}).get("fp", np.nan),
                        "tn*": r.get("cm_thr_star", {}).get("tn", np.nan), "fn*": r.get("cm_thr_star", {}).get("fn", np.nan),
                    })
                out_csv = os.path.join(tables_dir, f"mcq_{m}_perlayer_metrics.csv")
                pd.DataFrame(rows).to_csv(out_csv, index=False); print(f"[SAVE] {out_csv}")

        # === انتخاب لایه‌ها برای ویژوال ===
        if method == 'massmean':
            acc_map_for_pick = {li: acc for li, acc in per_layer_acc.items()}
            per_method_bests = {"massmean": best_layer} if best_layer is not None else {}
        else:
            acc_map_for_pick = _layer_metric_from_all_probes(probe_results, mode="max")
            per_method_bests = _best_layer_per_method(probe_results)

        common_layers = _spec_layers_without_best(acc_map_for_pick, args.viz_layers, len(Xte_layers))
        want_best = ("best" in [t.strip() for t in args.viz_layers.split(",") if t.strip()])

        layers_for_this_method = set(common_layers)
        if want_best and method in per_method_bests:
            layers_for_this_method.add(per_method_bests[method])

        for li in sorted(layers_for_this_method):
            Xtr = Xtr_layers[li]; Xte = Xte_layers[li]; yy = yte
            if method == "massmean":
                s = scores_for_probe("massmean", Xtr, ytr, Xte)
            else:
                s = probe_results[method][li]["s"]
            label_best = " (BEST)" if (method in per_method_bests and li == per_method_bests[method]) else ""
            title = f"MCQ | {method.upper()} | Layer {li}{label_best}"
            out_png = os.path.join(figs_dir, f"mcq_combined_{method}_layer_{li}.png")
            plot_combined_diagnostics(s, yy, title, out_png, show_kde=True, pca_Xte=Xte, pca_Xtr=Xtr, pca_yte=yte)  

        else:
            # ---------------------------- SINGLE ----------------------------
            pairs = build_pairs_single(model=model, tok=tokenizer, device=device, items=single_items, negatives_per=args.single_negatives_per)
            if not pairs:
                print("[Single] No single items found."); return

            train_pairs, test_pairs = split_train_test(pairs, test_ratio=args.test_ratio, seed=args.seed)

            
            Xtr_layers, ytr = _build_single_propmpt_feature(train_pairs, model, tokenizer)
            Xte_layers, yte = _build_single_propmpt_feature(test_pairs,  model, tokenizer)

            best_layer = None
            probe_results = {}

            if method == 'massmean':
                W = mass_mean_fit_from_pairs(train_pairs, model, tokenizer, single_alllayer_hiddens)
                per_layer_acc, best_layer, best_acc = eval_binary_from_pairs(test_pairs, W, model, tokenizer, single_alllayer_hiddens)
                out_csv = os.path.join(tables_dir, "massmean_single_perlayer_acc.csv")
                pd.DataFrame([{"layer": li, "acc": acc} for li, acc in per_layer_acc.items()]).to_csv(out_csv, index=False)
                plot_layer_acc_curve(per_layer_acc, f"Single Mass-Mean per-layer ({args.model})", os.path.join(figs_dir, "massmean_single_perlayer_acc.png"))
                print(f"[SAVE] {out_csv}")
            else:
                keys = sorted(Xtr_layers.keys())
                mid_li = keys[len(keys)//2]
                best_params = tune_probes_on_layer(Xtr_layers[mid_li], ytr, random_state=args.seed, method=method)
                print("[TUNE] best params:", best_params)
                probe_results = fit_eval_probes_per_layer(Xtr_layers, ytr, Xte_layers, yte, method=method, best_params=best_params)

                # ذخیره CSV
                for m, res in probe_results.items():
                    rows = []
                    for li, r in sorted(res.items()):
                        rows.append({
                            "layer": li, "acc": r.get("acc", np.nan), "auroc": r.get("auroc", np.nan),
                            "fisher": r.get("fisher", np.nan), "thr0": r.get("thr0", 0.0), "thr_star": r.get("thr_star", 0.0),
                            "tp0": r.get("cm_thr0", {}).get("tp", np.nan), "fp0": r.get("cm_thr0", {}).get("fp", np.nan),
                            "tn0": r.get("cm_thr0", {}).get("tn", np.nan), "fn0": r.get("cm_thr0", {}).get("fn", np.nan),
                            "tp*": r.get("cm_thr_star", {}).get("tp", np.nan), "fp*": r.get("cm_thr_star", {}).get("fp", np.nan),
                            "tn*": r.get("cm_thr_star", {}).get("tn", np.nan), "fn*": r.get("cm_thr_star", {}).get("fn", np.nan),
                        })
                    out_csv = os.path.join(tables_dir, f"single_{m}_perlayer_metrics.csv")
                    pd.DataFrame(rows).to_csv(out_csv, index=False); print(f"[SAVE] {out_csv}")

            # لایه‌های ویژوال
            if method == 'massmean':
                acc_map_for_pick = {li: acc for li, acc in per_layer_acc.items()}
                per_method_bests = {"massmean": best_layer} if best_layer is not None else {}
            else:
                acc_map_for_pick = _layer_metric_from_all_probes(probe_results, mode="max")
                per_method_bests = _best_layer_per_method(probe_results)

            common_layers = _spec_layers_without_best(acc_map_for_pick, args.viz_layers, len(Xte_layers))
            want_best = ("best" in [t.strip() for t in args.viz_layers.split(",") if t.strip()])

            layers_for_this_method = set(common_layers)
            if want_best and method in per_method_bests:
                layers_for_this_method.add(per_method_bests[method])

            for li in sorted(layers_for_this_method):
                Xtr = Xtr_layers[li]; Xte = Xte_layers[li]; yy = yte
                s = scores_for_probe("massmean", Xtr, ytr, Xte) if method == "massmean" else probe_results[method][li]["s"]
                label_best = " (BEST)" if (method in per_method_bests and li == per_method_bests[method]) else ""
                title = f"SINGLE | {method.upper()} | Layer {li}{label_best}"
                out_png = os.path.join(figs_dir, f"single_{method}_layer_{li}.png")
                plot_combined_diagnostics(s, yy, title, out_png, show_kde=True, pca_Xte=Xte, pca_Xtr=Xtr, pca_yte=yte)

if __name__ == "__main__":
    main()
