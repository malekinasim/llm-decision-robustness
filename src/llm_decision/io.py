import os
import json

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
            p = item.get("prompt", "")
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
                    raise ValueError(
                        f"{it.get('id','?')}: option {o!r} not single-token (len={len(ids)}). "
                        "Try leading space/casing to make it a single token."
                    )
                opt_ids.append(ids)
            it["_option_ids"] = opt_ids
            mcq_items.append(it)
        else:
            free_items.append(it)
    return mcq_items, 

