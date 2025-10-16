import re
import random
from typing import Tuple

# --- helpers ---------------------------------------------------------------

_ANSWER_SPLITS = [
    r"(?i)\banswer\s*:"
]

def _split_prefix_suffix(prompt: str) -> Tuple[str, str]:
    """
    Split prompt into (prefix, tail) where 'tail' starts at 'Answer:' (or 'پاسخ:') if present.
    If not present, tail == '' and whole prompt is prefix.
    """
    for pat in _ANSWER_SPLITS:
        m = re.search(pat, prompt)
        if m:
            return prompt[:m.start()], prompt[m.start():]
    return prompt, ""


def _normalize_space(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s)


# --- 1) Ask Twice ----------------------------------------------------------

def ask_twice(prompt: str) -> str:
    """
    Duplicate the question part once (before 'Answer:'), preserving the tail unchanged.
    """
    prefix, tail = _split_prefix_suffix(prompt)
    a = _normalize_space(prefix.strip())
    if not a:
        return prompt
    doubled = a + "\n\n" + a  # two lines: ask twice
    return f"{doubled}\n{tail}" if tail else doubled


# --- 2) Repeat Neutral Phrase ---------------------------------------------

_NEUTRAL_PHRASES = [
    "Please answer briefly.",
    "Note: This is the same question.",
    "Kindly provide a concise answer.",
]

def repeat_neutral(prompt: str, k: int = 1) -> str:
    """
    Append k neutral filler phrases to the question part (before 'Answer:').
    """
    prefix, tail = _split_prefix_suffix(prompt)
    a = prefix.rstrip()
    if k <= 0 or not a:
        return prompt

    # choose a fixed phrase to avoid semantic drift; keep it simple and short
    phrase = " " + _NEUTRAL_PHRASES[0]
    repeated = a + (phrase * k)
    return f"{repeated}\n{tail}" if tail else repeated


# --- 3) Small Typo ---------------------------------------------------------

def typo_small(prompt: str) -> str:
    """
    Introduce a single, small typo in the question part:
    - pick a word (>=4 chars, contains a letter, not purely numeric)
    - swap two adjacent inner characters (or drop one if swap not possible)
    Deterministic per prompt (seeded by hash) so runs are repeatable.
    """
    prefix, tail = _split_prefix_suffix(prompt)
    text = prefix

    # find candidate words (unicode-aware)
    # \w includes letters/digits/underscore with Unicode; filter to those with a letter
    words = list(re.finditer(r"\w+", text, flags=re.UNICODE))

    # filter: length >= 4, contains a letter, not purely digits
    cand = []
    for m in words:
        w = m.group(0)
        if len(w) >= 4 and not w.isdigit() and re.search(r"\p{L}", w, flags=re.UNICODE) if hasattr(re, "UNICODE") else re.search(r"[A-Za-zآ-ی]", w):
            cand.append(m)
    if not cand:
        return prompt  # nothing to do

    # deterministic selection per prompt
    rnd = random.Random(hash(prompt) & 0xffffffff)
    m = rnd.choice(cand)
    w = m.group(0)

    # choose inner position
    if len(w) >= 5:
        i = rnd.randrange(1, len(w) - 1)  # inner index
        # prefer swap with next if possible
        if i + 1 < len(w) - 0:
            w2 = w[:i] + w[i+1] + w[i] + w[i+2:]
        else:
            w2 = w[:i] + w[i+1:]
    else:
        # length == 4 -> drop one inner char
        i = 1
        w2 = w[:i] + w[i+1:]

    # rebuild
    new_prefix = text[:m.start()] + w2 + text[m.end():]
    return f"{new_prefix}{tail}" if tail else new_prefix
