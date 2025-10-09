class TunedDiag:
    """
    Simple diagonal tuned lens:
        x' = gamma[l] ⊙ x + beta[l]
    Optional per-layer gain on logits:
        z'_tuned = alpha[l] * (x' @ W_U)
    """
    def __init__(self, gamma=None, beta=None, alpha=None):
        self.gamma  = gamma or {}   # dict[int] -> torch.Tensor[d]
        self.beta   = beta  or {}   # dict[int] -> torch.Tensor[d]
        self._alpha = alpha or {}   # dict[int] -> float

    @staticmethod
    def from_json(path, device):
        import json, torch
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # {"layers": {"12": {"gamma":[...], "beta":[...], "alpha": 0.97}, ...}}
        gamma, beta, alpha = {}, {}, {}
        for k, v in data.get("layers", {}).items():
            l = int(k)
            if "gamma" in v:
                gamma[l] = torch.tensor(v["gamma"], dtype=torch.float32, device=device)
            if "beta" in v:
                beta[l]  = torch.tensor(v["beta"],  dtype=torch.float32, device=device)
            if "alpha" in v:
                alpha[l] = float(v["alpha"])
        return TunedDiag(gamma=gamma, beta=beta, alpha=alpha)

    def apply_x(self, l, x):  # x: [d] -> returns transformed hidden
        g = self.gamma.get(l)
        b = self.beta.get(l)
        if g is not None:
            x = x * g
        if b is not None:
            x = x + b
        return x
    def alpha(self, l):     # returns float or None
        return self._alpha.get(l, None)
