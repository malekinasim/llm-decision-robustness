# scripts/viz_linear_prob.py
import sys, os, argparse, numpy as np, pandas as pd
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from src.metrics import roc_auc
from src.feature_cache import load_feature_cache
from src.io import ensure_dirs
from src.linear_probes import fit_eval_probes_per_layer, tune_probes_on_layer, scores_for_probe
from src.viz import plot_combined_diagnostics, plot_layer_acc_curve
from src.mass_mean import (                                                
    mass_mean_eval_per_layer_feature,acc_question,mean_margin
)



def _pick_best_layer(res_dict: dict, metric: str) -> tuple[int, dict[int, float]]:
    """
    res_dict: {layer_idx -> {'acc':..., 'auroc':..., 'mean_margin':..., 'fisher':...}}
    metric:   one of {'auroc','acc','mean_margin','fisher'}
    returns: (best_li, values_per_layer)
    """
    vals = {}
    for li, r in res_dict.items():
        if metric == "acc":
            vals[li] = r.get("acc", float("-inf"))
        elif metric == "mean_margin":
            vals[li] = r.get("mean_margin", float("-inf"))
        elif metric == "fisher":
            vals[li] = r.get("fisher", float("-inf"))
        else:
            vals[li] = r.get("auroc", float("-inf"))  # default
    # اگر مقدار برابر بود، لایه عمیق‌تر ترجیح داده شود
    best_li = max(vals.keys(), key=lambda k: (vals[k], k))
    return int(best_li), vals

def _spec_layers(spec, L_total, best_map=None, available_layers=None):
    """
    spec: 'best,first,mid,last,10,...'
    best_map: dict[method] -> int | Iterable[int]   (BESTهای هر روش)
    """
    tokens = [t.strip().lower() for t in (spec or "").split(",") if t.strip()]
    valid = set(range(int(L_total))) if available_layers is None else {int(li) for li in available_layers}
    tag2li = {"first": 0, "mid": int(L_total)//2, "last": int(L_total)-1}

    out = set()
    for t in tokens:
        if t in tag2li:
            li = tag2li[t]
            if li in valid: out.add(li)
        elif t == "best" and best_map:
            for m, v in best_map.items():
                if isinstance(v, (list, tuple, set)):
                    for li in v:
                        li = int(li)
                        if li in valid: out.add(li)
                else:
                    li = int(v)
                    if li in valid: out.add(li)
        else:
            try:
                li = int(t)
                if li in valid: out.add(li)
            except ValueError:
                pass
    return sorted(int(li) for li in out)


def main():
    ap = argparse.ArgumentParser("Phase-2: run probes & plots from cached features")
    ap.add_argument("--task", choices=["mcq","single"], required=True)
    ap.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    ap.add_argument("--out_root", default=str(REPO_ROOT / "out"))
    ap.add_argument("--methods",  default="lda,logreg,linsvm",
                    help="comma list among: massmean,lda,logreg,linsvm")
    ap.add_argument("--viz_layers", default="best,first,mid,last")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
    "--best_by",
    default="auroc",
    help="Comma list of metrics to pick BEST layers by (e.g., 'auroc,acc,mean_margin,fisher')"
    )
    args = ap.parse_args()

    model_path = args.model.replace("/", "__")
    cache_dir  = os.path.join(args.out_root, "features", model_path, args.task)
    tables_dir = os.path.join(args.out_root, "reports", model_path, "tables",  args.task)
    figs_dir   = os.path.join(args.out_root, "reports", model_path, "figures", args.task)
    ensure_dirs(tables_dir, figs_dir)

    # Load cached features
    tr_npz = Path(os.path.join(cache_dir, "train.npz"))
    te_npz = Path(os.path.join(cache_dir, "test.npz"))
    Xtr_layers, ytr, qtr = load_feature_cache(tr_npz)
    Xte_layers, yte, qte = load_feature_cache(te_npz)
    L = len(Xtr_layers)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # --- tune on a mid layer (for learned probes) ---
    mid_li = sorted(Xtr_layers.keys())[L//2]
    X_tune, y_tune = Xtr_layers[mid_li], ytr
    tuned = {}
    for m in methods:
        if m in {"lda","logreg","linsvm"}:
            tuned[m] = tune_probes_on_layer(X_tune, y_tune, random_state=args.seed, method=m)
        else:
            tuned[m] = None

    # --- fit & eval across layers ---
    res_all = {}
    for m in methods:
        if m == "massmean":
                # ---- MASS-MEAN: fit once on TRAIN, eval on TEST (per layer) ----
            W = mass_mean_eval_per_layer_feature(Xtr_layers, ytr)
            res = {}
            for li, (w, b) in W.items():
                Xte = Xte_layers[li]
                s   = Xte @ w + b
                acc = acc_question(s, yte, qte)
                try:
                    au  = roc_auc(yte,s)
                except Exception:
                    au  = np.nan
                mm  = mean_margin(s, yte, qte)
                res[li] = {"acc": acc, "auroc": au, "mean_margin": mm, "scores": s}
            res_all[m] = res
        else:
            res_all[m] = fit_eval_probes_per_layer(
            Xtr_layers, ytr, Xte_layers, yte,
            best_params=tuned[m], method=m)

    for m, res in res_all.items():
        for li, r in res.items():
            s = r.get("scores")
            if s is None: 
                continue
            accQ = acc_question(s, yte, qte)    # Argmax داخل سؤال
            mm   = mean_margin(s, yte, qte)     # gold - best wrong
            r["acc_question"] = accQ
            r["mean_margin"]  = mm
            r["acc"] = accQ  

    # Save per-layer tables
    for m, res in res_all.items():
        rows = []
        for li, r in sorted(res.items()):
            rows.append({
                "layer": li,
                "acc":   r.get("acc",   np.nan),         
                "auroc": r.get("auroc", np.nan),
                "mean_margin": r.get("mean_margin", np.nan),
                "fisher":r.get("fisher",np.nan),
                "thr0":  r.get("thr0",  0.0),
                "thr*":  r.get("thr_star",0.0),
                "tp0":   r.get("cm_thr0", {}).get("tp", np.nan),
                "fp0":   r.get("cm_thr0", {}).get("fp", np.nan),
                "tn0":   r.get("cm_thr0", {}).get("tn", np.nan),
                "fn0":   r.get("cm_thr0", {}).get("fn", np.nan),
            })
        out_csv = os.path.join(tables_dir, f"{args.task}_{m}_perlayer_metrics.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("[SAVE]", out_csv)

    # --- pick layers for combined diagnostics (and PCA panel) ---
    # map best layer per method by AUROC


    best_by_list = [s.strip().lower() for s in args.best_by.split(",") if s.strip()]

    # 4-الف) نگاشت per-metric: metric -> {method -> best_layer}
    best_by_map = {mb: {} for mb in best_by_list}
    # 4-ب) نگاشت per-method از «اتحاد» BESTها: method -> set(layers)
    best_union = {m: set() for m in res_all.keys()}

    best_map = {}
    for m, res in res_all.items():
        if not res: 
            continue
        for mb in best_by_list:
            best_li, _ = _pick_best_layer(res, metric=mb)
            best_by_map[mb][m] = best_li
            best_union[m].add(best_li)

    for mb, mp in best_by_map.items():
        print(f"[best_by={mb}] " + ", ".join(f"{m}:{li}" for m, li in mp.items()))

    # 4-ج) لایه‌های موجود و لایه‌های خواسته‌شده (union of BESTs وقتی 'best' در spec هست)
    available_layers = sorted({int(li) for res in res_all.values() for li in res.keys()})
    want_layers = _spec_layers(args.viz_layers, L, best_map=best_union, available_layers=available_layers)
    print("want_layers:", want_layers)

    for m in methods:
        if m not in res_all: 
            continue
        for li in want_layers:
            if li not in res_all[m]:
                # print(f"[warn] {m}: missing layer {li}")
                continue

            layer_res = res_all[m][li]
            s = layer_res.get('scores', layer_res.get('score'))

            # کدام متریک‌ها این لایه را «BEST» دانسته‌اند؟
            badges = [mb for mb in best_by_list if best_by_map.get(mb, {}).get(m) == li]
            title = f"{args.task.upper()} | {m.upper()} | Layer {li}"
            if badges:
                title += " (BEST by " + ",".join(b.upper() for b in badges) + ")"

            out_png = os.path.join(figs_dir, f"{args.task}_{m}_layer_{li}.png")
            plot_combined_diagnostics(
                s, yte, title, out_png,
                show_kde=True,
                # PCA: fit روی Train، نمایش Test (upcast برای SVD)
                pca_Xtr=Xtr_layers[li].astype(np.float32, copy=False),
                pca_Xte=Xte_layers[li].astype(np.float32, copy=False),
                pca_yte=yte
            )

    # optional: an ACC curve using best (e.g., AUROC) per layer per-method
    for m, res in res_all.items():
        acc_map = {li: r.get("acc", np.nan) for li, r in res.items()}
        out_acc = os.path.join(figs_dir, f"{args.task}_{m}_acc_curve.png")
        plot_layer_acc_curve(acc_map, f"{args.task.upper()} {m.upper()} per-layer ACC ({args.model})", out_acc)
    
   
    rows = []
    for mb, mp in best_by_map.items():
        for m, li in mp.items():
            rows.append({"metric": mb, "method": m, "best_layer": int(li)})
    pd.DataFrame(rows).to_csv(
        os.path.join(tables_dir, f"{args.task}_best_layers_by_metrics.csv"),
        index=False
    )

if __name__ == "__main__":
    main()
