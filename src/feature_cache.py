# src/feature_cache.py
import os, numpy as np

def save_feature_cache(path, X_layers, y, qids=None, dtype=np.float16):
    """Save per-layer features into a single NPZ.
       X_layers: dict[layer_idx] -> np.ndarray [N, D]
       y: (N,) labels (MCQ: gold=1, wrong=0; Single: gold=1, decoy=0)
       qids: (N,) optional question-ids (needed for MCQ Accuracy@Question)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pack = {"y": np.asarray(y, dtype=np.int64)}
    if qids is not None:
        pack["qids"] = np.asarray(qids)
    for li, X in X_layers.items():
        pack[f"X_l{li}"] = np.asarray(X, dtype=dtype)
    np.savez_compressed(path, **pack)

def load_feature_cache(path):
    """Return (X_layers, y, qids)."""
    z = np.load(path, allow_pickle=True)
    keys = [k for k in z.files if k.startswith("X_l")]
    layers = sorted(int(k.split("X_l")[1]) for k in keys)
    X_layers = {li: z[f"X_l{li}"] for li in layers}
    y = z["y"].astype(np.int64)
    qids = z["qids"] if "qids" in z.files else None
    return X_layers, y, qids
