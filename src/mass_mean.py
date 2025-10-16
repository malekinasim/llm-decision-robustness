import torch
import numpy as np

def     mass_mean_eval_per_layer_feature(Xtr_layers, ytr):
    """Return dict: layer -> (w, b) using difference-of-means."""
    W = {}
    ytr = np.asarray(ytr).astype(int)
    for li, X in Xtr_layers.items():
        pos = X[ytr == 1]
        neg = X[ytr == 0]
        if len(pos) == 0 or len(neg) == 0:
            # degenerate: no both classes; fall back to zeros
            w = np.zeros(X.shape[1], dtype=np.float32); b = 0.0
        else:
            mu_pos = pos.mean(axis=0)
            mu_neg = neg.mean(axis=0)
            w = (mu_pos - mu_neg).astype(np.float32)
            b = -0.5 * float(w @ (mu_pos + mu_neg))
        W[li] = (w, b)
    return W
def mass_mean_fit_per_layer(train_items, model, tokenizer, get_layer_hiddens_fn, pos=-1):
    """
    Returns dict: layer_idx -> (w vector as torch.Tensor on CPU)
    train_items: list of MCQ dicts: {"question","options","answer"}
    get_layer_hiddens_fn: function like mcq_alllayer_hiddens
    """
    # Accumulate per-layer positives/negatives
    pos_acc = {}  # layer -> list[tensor]
    neg_acc = {}
    for it in train_items:
        prompt = it["question"]; options = it["options"]; gold = it.get("answer", None)
        if (gold is None) or (gold not in options):  # skip if no label
            continue
        layer_dicts = get_layer_hiddens_fn(model, tokenizer, prompt, options,  skip_embedding=True)
        for li, od in enumerate(layer_dicts):
            for opt, h in od.items():
                (pos_acc.setdefault(li, []) if opt == gold else neg_acc.setdefault(li, [])).append(h)
    # compute w = mu_T - mu_F per layer
    W = {}
    for li in pos_acc.keys():
        if li not in neg_acc or len(pos_acc[li]) == 0 or len(neg_acc[li]) == 0:
            continue
        mu_T = torch.stack(pos_acc[li]).mean(dim=0)
        mu_F = torch.stack(neg_acc[li]).mean(dim=0)
        w = (mu_T - mu_F).detach().cpu()
        # normalize optional (often helps)
        w = w / (w.norm() + 1e-8)
        W[li] = w
    return W

def acc_question(scores, y, qids):
    """Accuracy@Question (MCQ). If qids is None, falls back to threshold @0."""
    y = np.asarray(y).astype(int)
    if qids is None:
        yp = (scores >= 0.0).astype(int)
        return float((yp == y).mean())
    acc = []
    uq = np.unique(qids)
    for q in uq:
        idx = (qids == q)
        s_q, y_q = scores[idx], y[idx]
        if s_q.size == 0: continue
        acc.append(int(y_q[np.argmax(s_q)] == 1))
    return float(np.mean(acc)) if acc else 0.0

def mean_margin(scores, y, qids):
    """Mean (gold - best wrong) margin per question (MCQ). If qids None, use class margins."""
    y = np.asarray(y).astype(int)
    if qids is None:
        # simple binary margin (gold mean - wrong max)
        return float(scores[y==1].mean() - (scores[y==0].max() if (y==0).any() else 0.0))
    margins = []
    for q in np.unique(qids):
        idx = (qids == q)
        s_q, y_q = scores[idx], y[idx]
        if (y_q==1).sum()==1 and (y_q==0).sum()>=1:
            margins.append(float(s_q[y_q==1][0] - np.max(s_q[y_q==0])))
    return float(np.mean(margins)) if margins else np.nan
def mass_mean_eval_per_layer(test_items, W, model, tokenizer, get_layer_hiddens_fn, pos=-1):
    """
    Given learned W per layer, returns:
      per_layer_acc: dict[layer_idx] -> accuracy over test MCQs
      best_layer, best_acc
    """
    per_layer_right = {li: 0 for li in W.keys()}
    per_layer_total = {li: 0 for li in W.keys()}
    for it in test_items:
        prompt = it["question"]; options = it["options"]; gold = it.get("answer", None)
        if (gold is None) or (gold not in options):
            continue
        layer_dicts = get_layer_hiddens_fn(model, tokenizer, prompt, options,  skip_embedding=True)
        for li, od in enumerate(layer_dicts):
            if li not in W: 
                continue
            w = W[li]
            # score each option by dot(w, h_opt)
            best_opt, best_s = None, -1e30
            for opt, h in od.items():
                s = float(torch.dot(w, h))
                if s > best_s:
                    best_s, best_opt = s, opt
            per_layer_total[li] += 1
            per_layer_right[li] += int(best_opt == gold)
    per_layer_acc = {li: (per_layer_right[li] / per_layer_total[li] if per_layer_total[li] else 0.0)
                     for li in per_layer_total}
    # pick best
    best_layer, best_acc = None, -1.0
    for li, acc in per_layer_acc.items():
        if acc > best_acc:
            best_layer, best_acc = li, acc
    return per_layer_acc, best_layer, best_acc


def mass_mean_fit_from_pairs(
    pairs, model, tokenizer, get_single_hiddens_fn, skip_embedding=True
):
    """
    pairs: list of (prompt, completion, label, group_id)
    Returns: dict[layer_idx] -> w (torch.Tensor on CPU, normalized)
    """
    # batch forward in chunks to keep memory sane
    BATCH = 32
    pos_per_layer, neg_per_layer = {}, {}
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i+BATCH]
        prompts = [x[0] for x in chunk]
        comps   = [x[1] for x in chunk]
        labels = [x[2] for x in chunk]
        # Robustly coerce every label to a scalar int (handles array([1]), lists, numpy scalars, etc.)
        def _to_int_label(l):
            import numpy as np
            # numpy scalar -> python scalar
            if hasattr(l, "item") and not isinstance(l, (list, tuple)):
                try: l = l.item()
                except Exception: pass
            # sequence -> first element (define your policy)
            if isinstance(l, (list, tuple, np.ndarray)):
                if len(l) == 0: return 0
                l = l[0]
                if hasattr(l, "item"):
                    try: l = l.item()
                    except Exception: pass
            try:
                return int(l)
            except Exception:   
                return int(bool(l))

        labels_arr = np.array([_to_int_label(l) for l in labels], dtype=np.int64)

           
        comps = [str(c) for c in comps]
        prompts = [str(p) for p in prompts]
        layer_vecs = get_single_hiddens_fn(
            model, tokenizer, prompts, comps, skip_embedding=skip_embedding
        )  # List[L] of [b,D]
        L = len(layer_vecs)
        for li in range(L):
            V = layer_vecs[li]  # [b,D] on CPU
            for j in range(len(labels_arr)):
                lab=int(labels_arr[j])
                if lab == 1:
                    pos_per_layer.setdefault(li, []).append(V[j])
                else:
                    neg_per_layer.setdefault(li, []).append(V[j])
    W = {}
    for li in pos_per_layer.keys():
        if li not in neg_per_layer or len(pos_per_layer[li]) == 0 or len(neg_per_layer[li]) == 0:
            continue
        mu_T = torch.stack(pos_per_layer[li]).mean(dim=0)
        mu_F = torch.stack(neg_per_layer[li]).mean(dim=0)
        w = mu_T - mu_F
        w = w / (w.norm() + 1e-8)
        W[li] = w.detach().cpu()
    return W