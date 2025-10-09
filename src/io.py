import os
import json
import pandas as pd

def ensure_dirs(rport_table_path,report_figur_path):
    os.makedirs(rport_table_path,exist_ok=True)
    os.makedirs(report_figur_path,exist_ok=True)

def load_prompts_from_json(path):
    if not os.path.exists(path):
        raise ValueError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts, labels = [], []

    def norm_label(t):
        if not t:
            return "unknown"
        t = str(t).strip().lower()
        if t.startswith("dec"):
            return "decision"
        if t.startswith("con"):
            return "control"
        return t

    for i, item in enumerate(data):
        if isinstance(item, dict):
            p = item.get("question", "")
            t = item.get("type", "")
        else:
            p, t = str(item), ""
        p = p.strip()
        if not p:
            continue
        prompts.append(p)
        labels.append(norm_label(t))
    if not prompts:
        raise ValueError("No prompts found in JSON.")
    return prompts, labels

def load_prompts_with_options(path, tokenizer, require_single_token=True):
    """
    Loads a mixed dataset:
      - MCQ items: have 'options' (and optional 'gold'), go to mcq_items
      - Single items: {'task':'single', 'prompt', 'gold'} go to free_items
      - Others (Control/Decision without options) also go to free_items
    Each MCQ is validated for single-token options when 'require_single_token' is True.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mcq_items, free_items = [], []
    for it in data:
        task = it.get("task") or ("mcq" if "options" in it else "free")
        it["task"] = task
        it.setdefault("pos", -1)

        if task == "mcq":
            opts = it.get("options", [])
            if not opts:
                raise ValueError(f"{it.get('id','?')}: 'options' required for mcq.")
            opt_ids = []
            for o in opts:
                ids = tokenizer.encode(o, add_special_tokens=False)
                if require_single_token and len(ids) != 1:
                    print(
                        f"{it.get('id','?')}: option {o!r} not single-token (len={len(ids)}). "
                        "Try leading space/casing to make it a single token."
                    )
                    continue
                opt_ids.append(ids)
            if(len(opt_ids)>0):
                it["_option_ids"] = opt_ids
                mcq_items.append(it)
        else:
            free_items.append(it)
    return mcq_items, free_items



# =============== Save (CSV ) ===============
def save_CSV_layers_MCQ_Margins(res, options, out_dir="out_doc", fname="mcq_perlayer_margins.csv"):
    os.makedirs(out_dir, exist_ok=True)
    raw_scores, raw_winners, raw_m12, raw_gold = res["raw"]
    tuned_part = res["tuned"]

    rows = []
    L = len(raw_scores)
    for li in range(L):
        row = {
            "layer": li,
            "winner_raw":  raw_winners[li],
            "margin_top1_top2_raw": raw_m12[li],
            "gold_margin_opts_raw": (raw_gold[li] if raw_gold else None),
        }
        if tuned_part is not None and li < len(tuned_part[0]):
            tuned_scores, tuned_winners, tuned_m12, tuned_gold = tuned_part
            row.update({
                "winner_tuned": tuned_winners[li],
                "margin_top1_top2_tuned": tuned_m12[li],
                "gold_margin_opts_tuned": (tuned_gold[li] if tuned_gold else None),
            })
        for opt in options:
            row[f"logit_raw[{opt}]"] = raw_scores[li][opt]
            if tuned_part is not None and li < len(tuned_part[0]):
                row[f"logit_tuned[{opt}]"] = tuned_scores[li][opt]
        rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)


def save_csv_margins(margins, out_dir='out_doc', fname='margins_per_layer.csv'):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    def add_rows(branch_name, data_dict):
        if not data_dict: return
        L = max((len(v) for v in data_dict.values()), default=0)
        for i in range(L):
            row = {"layer": i, "branch": branch_name}
            for k, seq in data_dict.items():
                if i < len(seq): row[k] = seq[i]
            rows.append(row)
    for space in ("full", "opts"):
        if space in margins:
            for ver in ("raw","tuned"):
                if ver in margins[space]:
                    add_rows(f"{space}_{ver}", margins[space][ver])
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, fname), index=False)