import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support

EPS = 1e-8

def fisher_score(scores: np.ndarray, y: np.ndarray) -> float:
    """Fisher = ((mu_pos - mu_neg)^2) / (var_pos + var_neg) on 1D scores."""
    sp = scores[y == 1]
    sn = scores[y == 0]
    if sp.size == 0 or sn.size == 0:
        return float("nan")
    num = (sp.mean() - sn.mean())**2
    den = sp.var() + sn.var() + EPS
    return float(num / den)

def roc_auc( y: np.ndarray,scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    return float(auc(fpr, tpr))

def youden_optimal_threshold(scores: np.ndarray, y: np.ndarray):
    """Return threshold maximizing Youden’s J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y, scores)
    if len(thr) == 0:
        return 0.0, 0.0
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(j[k])

def confusion_from_scores(scores: np.ndarray, y: np.ndarray, thr: float = 0.0):
    """Confusion + standard metrics at a given threshold."""
    y_hat = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0,1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                precision=float(prec), recall=float(rec), f1=float(f1), acc=float(acc))
