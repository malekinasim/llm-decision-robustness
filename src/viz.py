# imports (یک‌بار کافی است)
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix, accuracy_score
)

# ==== Global palette (consistent colors) ====
COLOR_POS     = "#1fb42b"  # Green  - y=1 (gold)
COLOR_NEG     = "#ee5151"  # red- y=0 (wrong)
COLOR_THR0    = "#5a5759"  # grey
COLOR_THRSTAR = "#0f70b6"  # blue (می‌تونی سبز بگذاری اگر تمایز می‌خواهی)

CMAP_CM_THR0    = "Blues"     # confusion @ thr=0
CMAP_CM_THRSTAR = "Blues"   # confusion @ thr*


def plot_margins(margins, path='fig', fname='margins_per_layer.png', title='Margins per Layer'):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8,4))
    if "full" in margins:
        m = margins["full"]
        plt.plot(m["raw"]["top1_top2_full"], label="top1-top2 full (raw)")
        if "gold_full" in m["raw"]:
            plt.plot(m["raw"]["gold_full"], label="gold full (raw)")
        if "tuned" in m:
            plt.plot(m["tuned"]["top1_top2_full"], label="top1-top2 full (tuned)")
            if "gold_full" in m["tuned"]:
                plt.plot(m["tuned"]["gold_full"], label="gold full (tuned)")
    if "opts" in margins:
        m = margins["opts"]
        plt.plot(m["raw"]["top1_top2_opts"], label="top1-top2 opts (raw)")
        if "gold_opts" in m["raw"]:
            plt.plot(m["raw"]["gold_opts"], label="gold opts (raw)")
        if "tuned" in m:
            plt.plot(m["tuned"]["top1_top2_opts"], label="top1-top2 opts (tuned)")
            if "gold_opts" in m["tuned"]:
                plt.plot(m["tuned"]["gold_opts"], label="gold opts (tuned)")
    plt.xlabel("Layer index (0=embedding unless skipped)")
    plt.ylabel("Margin"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(path, fname), dpi=200); plt.close()



def plot_MCQ_Margin(res, out_png="fig/mcq_margins_per_layer.png",
                               title="Per-layer margins (raw vs tuned)"):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    raw_scores, _, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]

    plt.figure(figsize=(8,4))
    plt.plot(raw_m12, label="top1-top2 (raw)")
    if raw_gold:
        plt.plot(raw_gold, label="gold-margin (raw)")
    if tuned_part is not None:
        _, _, tuned_m12, tuned_gold = tuned_part
        plt.plot(tuned_m12, label="top1-top2 (tuned)")
        if tuned_gold:
            plt.plot(tuned_gold, label="gold-margin (tuned)")
    plt.xlabel("Layer index (0 = first after embedding if skipped)")
    plt.ylabel("Margin")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



def plot_layer_acc_curve(per_layer_acc: dict, title: str, out_path: str):
    """
    Plot a simple layer->accuracy curve.
    per_layer_acc: dict {layer_index(int): acc(float)}
    """
    layers = sorted(per_layer_acc.keys())
    accs = [per_layer_acc[l] for l in layers]
    plt.figure(figsize=(7,4))
    plt.plot(layers, accs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(alpha=0.25)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_sep_pca_lda(X: np.ndarray, y: np.ndarray, title: str, out_path: str):
    """
    Visualize separation for a single layer.
    X: [N, D] layer features; y in {0,1} (1=gold/positive, 0=wrong/negative)
    Draws PCA-2D and LDA-1D (padded to 2D).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # PCA 2D
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(Xs)

    # LDA 1D (pad to 2D for consistent plotting)
    try:
        X_lda1 = LDA(n_components=1).fit_transform(Xs, y)  # [N,1]
        X_lda = np.c_[X_lda1, np.zeros_like(X_lda1)]
    except Exception:
        X_lda = np.zeros((Xs.shape[0], 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _scatter(ax, Z, name):
        ax.scatter(Z[y==1, 0], Z[y==1, 1], s=12, alpha=0.75, label="Positive / Gold", marker="o")
        ax.scatter(Z[y==0, 0], Z[y==0, 1], s=12, alpha=0.75, label="Negative / Wrong", marker="x")
        ax.set_title(name); ax.grid(alpha=.2)

    _scatter(axes[0], X_pca, "PCA-2D")
    _scatter(axes[1], X_lda, "LDA-1D (padded)")

    fig.suptitle(title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_from_scores(scores: np.ndarray, y: np.ndarray, title: str, out_path: str) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1], 'k--', alpha=.3)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return float(roc_auc)

def plot_scores_hist(s_pos: np.ndarray, s_neg: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7,4))
    bins = 40
    plt.hist(s_pos, bins=bins, alpha=0.6, label="Positive/Gold")
    plt.hist(s_neg, bins=bins, alpha=0.6, label="Negative/Wrong")
    plt.xlabel("w^T x (or classifier score)"); plt.ylabel("count")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def plot_confusion_matrix(cm, class_names, title, out_path):
    """
    cm: 2x2 numpy array [[TN, FP], [FN, TP]]
    class_names: e.g. ["Wrong/Negative", "Gold/Positive"]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticks([0,1]); ax.set_yticklabels(class_names)
    # values
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v)}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_score_density(scores_pos: np.ndarray, scores_neg: np.ndarray, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7,4))

    s_all = np.concatenate([scores_pos, scores_neg])
    lo, hi = np.percentile(s_all, [1, 99])
    grid = np.linspace(lo, hi, 400)[:, None]

    for s, label, color in [
        (scores_pos, "Positive/Gold", COLOR_POS),
        (scores_neg, "Negative/Wrong", COLOR_NEG),
    ]:
        if len(s) < 2:
            continue
        kde = KernelDensity(kernel='gaussian', bandwidth=(hi-lo)/30.0).fit(s[:, None])
        log_d = kde.score_samples(grid)
        plt.plot(grid[:,0], np.exp(log_d), label=label, color=color)

    plt.title(title); plt.xlabel("score s = w^T x + b"); plt.ylabel("density")
    plt.legend(); plt.grid(alpha=.2)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def fit_standardizer_on_train(Xtr):
    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return mu, sigma

def apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

def fit_pca2_on_train(Ztr):
    # SVD only on train (centered already توسط استانداردسازی)
    U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    W2 = Vt[:2].T          # projection matrix to 2D
    total_var = (S**2).sum()
    evr = (S[:2]**2) / total_var if total_var > 0 else np.array([np.nan, np.nan])
    return W2, evr

def transform_to_2d(Z, W2):
    return Z @ W2
def plot_pca2_test(Z2_test, y_test, title, out_png):
    plt.figure(figsize=(6,5))
    for cls in [0,1]:
        pts = Z2_test[y_test == cls]
        if pts.size == 0: continue
        lbl = "gold" if int(cls)==1 else "wrong"
        col = COLOR_POS if int(cls)==1 else COLOR_NEG
        plt.scatter(pts[:,0], pts[:,1], label=lbl, s=14, color=col)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close()

def pca_train_fit_test_plot(Xtr, Xte, yte, title, out_png):
    mu, sigma = fit_standardizer_on_train(Xtr)
    Ztr = apply_standardizer(Xtr, mu, sigma)
    Zte = apply_standardizer(Xte, mu, sigma)
    W2, evr = fit_pca2_on_train(Ztr)
    Z2_te = transform_to_2d(Zte, W2)
    plot_pca2_test(Z2_te, yte, f"{title} — PC1={evr[0]:.2f}, PC2={evr[1]:.2f}", out_png)
    return evr


def plot_combined_diagnostics(
    scores: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    show_kde: bool = True,
    bins: int = 40,
    # --- NEW: optional PCA inputs (fit on train, show test)
    pca_Xtr: np.ndarray = None,
    pca_Xte: np.ndarray = None,
    pca_yte: np.ndarray = None,
):
    """Make ONE compact figure with:
      - Score histogram + KDE (class-conditional)
      - ROC curve (AUROC)
      - PR curve (AP)
      - Confusion @ thr=0
      - Confusion @ thr* (Youden)
      - (NEW) PCA(2D) of TEST (PCA fitted on TRAIN)
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scores = np.asarray(scores).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    assert scores.shape[0] == y.shape[0]

    # ---- metrics (from probe scores)
    fpr, tpr, thr_roc = roc_curve(y, scores)
    auroc = auc(fpr, tpr)
    prec, rec, thr_pr = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    j = tpr - fpr
    k = int(np.argmax(j)) if len(j) else 0
    thr_star = float(thr_roc[k]) if len(thr_roc) else 0.0

    def cm_at(th):
        yp = (scores >= th).astype(int)
        cm = confusion_matrix(y, yp, labels=[0,1])
        acc = accuracy_score(y, yp)
        return cm, acc
    cm0, acc0 = cm_at(0.0)
    cm_star, acc_star = cm_at(thr_star)

    # ---- figure layout (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    (ax_hist, ax_roc, ax_pr), (ax_cm0, ax_cmstar, ax_pca) = axes

    # 1) histogram + KDE
    all_scores = scores
    lo, hi = np.percentile(all_scores, [1, 99]) if all_scores.size else (0, 1)
    lo, hi = (min(lo, 0.0), max(hi, 0.0))  # keep 0 visible
    bins_edges = np.linspace(lo, hi, bins)

    
    ax_hist.hist(all_scores[y==1], bins=bins_edges, density=True, alpha=0.6, label="Positive (y=1)",color=COLOR_POS)
    ax_hist.hist(all_scores[y==0], bins=bins_edges, density=True, alpha=0.6, label="Negative (y=0)",color=COLOR_NEG)
    if show_kde:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(lo, hi, 400)
            if (y==1).sum() >= 2:
                kde_pos = gaussian_kde(all_scores[y==1]); ax_hist.plot(xs, kde_pos(xs), lw=2, label="Pos KDE",color=COLOR_POS)
            if (y==0).sum() >= 2:
                kde_neg = gaussian_kde(all_scores[y==0]); ax_hist.plot(xs, kde_neg(xs), lw=2, label="Neg KDE",color=COLOR_NEG)
        except Exception:
            pass
    ax_hist.axvline(0.0, color=COLOR_THR0, linestyle="--", alpha=0.7, label="thr=0")
    ax_hist.axvline(thr_star, color=COLOR_THRSTAR, linestyle="-.", alpha=0.8, label=f"thr*={thr_star:.3f}")
    ax_hist.set_title("Score distribution")
    ax_hist.set_xlabel("score s = w^T x (+ b)"); ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize='small')

    # 2) ROC
    ax_roc.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax_roc.plot([0,1],[0,1],'--', alpha=.3)
    ax_roc.set_title("ROC"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc="lower right", fontsize="small"); ax_roc.grid(alpha=.2)

    # 3) PR
    baseline = y.mean() if y.size else 0.0
    ax_pr.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax_pr.hlines(baseline, 0, 1, linestyles='--', label=f"baseline={baseline:.3f}")
    ax_pr.set_title("Precision–Recall"); ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize="small"); ax_pr.grid(alpha=.2)

    # 4) CM @ thr=0
    im0 = ax_cm0.imshow(cm0, cmap=CMAP_CM_THR0)
    ax_cm0.set_title(f"Confusion @ thr=0 (ACC={acc0:.3f})")
    ax_cm0.set_xticks([0,1]); ax_cm0.set_yticks([0,1])
    ax_cm0.set_xticklabels(["Pred wrong","Pred gold"]); ax_cm0.set_yticklabels(["Rean wrong","Real gold"])
    for (i,j), v in np.ndenumerate(cm0):
        ax_cm0.text(j, i, int(v), ha="center", va="center",
                    color=("white" if v > cm0.max()/2 else "black"))
    fig.colorbar(im0, ax=ax_cm0, fraction=0.046, pad=0.04)

    # 5) CM @ thr*
    im1 = ax_cmstar.imshow(cm_star, cmap=CMAP_CM_THRSTAR)
    ax_cmstar.set_title(f"Confusion @ thr* (ACC={acc_star:.3f})")
    ax_cmstar.set_xticks([0,1]); ax_cmstar.set_yticks([0,1])
    ax_cmstar.set_xticklabels(["Pred wrong","Pred gold"]); ax_cmstar.set_yticklabels(["Real wrong","Real gold"])
    for (i,j), v in np.ndenumerate(cm_star):
        ax_cmstar.text(j, i, int(v), ha="center", va="center",
                       color=("white" if v > cm_star.max()/2 else "black"))
    fig.colorbar(im1, ax=ax_cmstar, fraction=0.046, pad=0.04)

    # 6) (NEW) PCA panel — fit on TRAIN, show TEST
    ax_pca.set_title("PCA (TEST; fit on TRAIN)")
    ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")
    if pca_Xtr is not None and pca_Xte is not None and pca_yte is not None:
        # standardize on train
        mu = pca_Xtr.mean(axis=0, keepdims=True)
        sd = pca_Xtr.std(axis=0, keepdims=True); sd[sd==0] = 1.0
        Ztr = (pca_Xtr - mu) / sd
        Zte = (pca_Xte - mu) / sd
        # PCA on TRAIN (SVD)
        U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
        W2 = Vt[:2].T
        tot = (S**2).sum()
        evr = (S[:2]**2)/tot if tot>0 else np.array([np.nan, np.nan])
        # project TEST
        Z2 = Zte @ W2
        for cls in np.unique(pca_yte):
            pts = Z2[pca_yte == cls]
            if pts.size == 0:
              continue
            color = COLOR_POS if cls == 1 else COLOR_NEG
            label = "gold" if cls == 1 else "wrong"
            ax_pca.scatter(pts[:, 0], pts[:, 1], s=12, label=label, color=color)
        ax_pca.legend(fontsize='small', loc="best")
        ax_pca.text(0.02, 0.98, f"PC1={evr[0]:.2f}, PC2={evr[1]:.2f}",
                    transform=ax_pca.transAxes, ha="left", va="top", fontsize=9)

    # Title + save
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)



def plot_combined_diagnostics_old(
    scores: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    show_kde: bool = True,
    bins: int = 40
):
    """Make ONE compact figure with:
      - Score histogram + KDE (class-conditional)
      - ROC curve (AUROC)
      - PR curve (AP)
      - Confusion @ thr=0
      - Confusion @ thr* (Youden)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Safety
    scores = np.asarray(scores).reshape(-1)
    y = np.asarray(y).astype(int).reshape(-1)
    assert scores.shape[0] == y.shape[0]

    # ---- metrics
    # ROC
    fpr, tpr, thr_roc = roc_curve(y, scores)
    auroc = auc(fpr, tpr)
    # PR
    prec, rec, thr_pr = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    # Youden-opt thr
    j = tpr - fpr
    k = int(np.argmax(j)) if len(j) else 0
    thr_star = float(thr_roc[k]) if len(thr_roc) else 0.0
    # Confusions
    def cm_at(th):
        yp = (scores >= th).astype(int)
        cm = confusion_matrix(y, yp, labels=[0,1])
        acc = accuracy_score(y, yp)
        return cm, acc
    cm0, acc0 = cm_at(0.0)
    cm_star, acc_star = cm_at(thr_star)

    # ---- figure layout (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    (ax_hist, ax_roc, ax_pr), (ax_cm0, ax_cmstar, ax_blank) = axes

    # 1) histogram + KDE
    all_scores = scores
    lo, hi = np.percentile(all_scores, [1, 99]) if all_scores.size else (0, 1)
    lo, hi = (min(lo, 0.0), max(hi, 0.0))  # keep 0 visible
    bins_edges = np.linspace(lo, hi, bins)
    ax_hist.hist(all_scores[y==1], bins=bins_edges, density=True, alpha=0.6, label="Positive (y=1)"
                 ,Color=COLOR_POS)
    ax_hist.hist(all_scores[y==0], bins=bins_edges, density=True, alpha=0.6, label="Negative (y=0)",
                 Color=COLOR_NEG)
    # KDE (optional)
    if show_kde:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(lo, hi, 400)
            if (y==1).sum() >= 2:
                kde_pos = gaussian_kde(all_scores[y==1]); ax_hist.plot(xs, kde_pos(xs), color='C0', lw=2, label="Pos KDE"
                ,Color=COLOR_POS)
            if (y==0).sum() >= 2:
                kde_neg = gaussian_kde(all_scores[y==0]); ax_hist.plot(xs, kde_neg(xs), color='C1', lw=2, label="Neg KDE",
                                                                       Color=COLOR_NEG)
        except Exception:
            pass
    # mark thr=0 and thr*
    ax_hist.axvline(0.0, Color=COLOR_THR0, linestyle="--", alpha=0.7, label="thr=0")
    ax_hist.axvline(thr_star, Color=COLOR_THRSTAR, linestyle="-.", alpha=0.8, label=f"thr*={thr_star:.3f}")
    ax_hist.set_title("Score distribution")
    ax_hist.set_xlabel("score s = w^T x (+ b)"); ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize='small')

    # 2) ROC
    ax_roc.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax_roc.plot([0,1],[0,1],'k--', alpha=.3)
    ax_roc.set_title("ROC"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.legend(loc="lower right", fontsize="small"); ax_roc.grid(alpha=.2)

    # 3) PR
    baseline = y.mean() if y.size else 0.0
    ax_pr.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax_pr.hlines(baseline, 0, 1, colors='gray', linestyles='--', label=f"baseline={baseline:.3f}")
    ax_pr.set_title("Precision–Recall"); ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize="small"); ax_pr.grid(alpha=.2)

    # 4) CM @ thr=0
    im0 = ax_cm0.imshow(cm0, cmap=CMAP_CM_THR0)
    ax_cm0.set_title(f"Confusion @ thr=0 (ACC={acc0:.3f})")
    ax_cm0.set_xticks([0,1]); ax_cm0.set_yticks([0,1])
    ax_cm0.set_xticklabels(["Pred wrong","Pred gold"]);
    ax_cm0.set_yticklabels(["Real wrong","Real gold"])
    for (i,j), v in np.ndenumerate(cm0):
        ax_cm0.text(j, i, int(v), ha="center", va="center",
                    color=("white" if v > cm0.max()/2 else "black"))
    fig.colorbar(im0, ax=ax_cm0, fraction=0.046, pad=0.04)

    # 5) CM @ thr*
    im1 = ax_cmstar.imshow(cm_star, cmap=CMAP_CM_THRSTAR)
    ax_cmstar.set_title(f"Confusion @ thr* (ACC={acc_star:.3f})")
    ax_cmstar.set_xticks([0,1]); ax_cmstar.set_yticks([0,1])
    ax_cmstar.set_xticklabels(["Pred wrong","Pred gold"]); 
    ax_cmstar.set_yticklabels(["Real wrong","Real gold"])
    for (i,j), v in np.ndenumerate(cm_star):
        ax_cmstar.text(j, i, int(v), ha="center", va="center",
                       color=("white" if v > cm_star.max()/2 else "black"))
    fig.colorbar(im1, ax=ax_cmstar, fraction=0.046, pad=0.04)

    # 6) blank / text box (optional)
    ax_blank.axis("off")
    ax_blank.text(0.0, 0.9, f"thr*= {thr_star:.3f}", fontsize=10)
    ax_blank.text(0.0, 0.75, f"ACC@0= {acc0:.3f}", fontsize=10)
    ax_blank.text(0.0, 0.60, f"ACC@thr*= {acc_star:.3f}", fontsize=10)
    ax_blank.text(0.0, 0.45, f"AUROC= {auroc:.3f}", fontsize=10)
    ax_blank.text(0.0, 0.30, f"AP= {ap:.3f}", fontsize=10)

    # Title + save
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

