
import numpy as np
from .filters import GaussianBilateral

class GaussianBilateralForward:
    def __init__(self, steps=12, mode="hybrid"):
        self.steps = steps
        self.mode = mode

    def run(self, x0):
        """
        Returns a list [x_T, x_{T-1}, ..., x_0] of length steps+1.
        """
        return GaussianBilateral(x0, steps=self.steps, mode=self.mode)
    
# === Cosine-schedule TRUE noise forward (x0 -> ... -> noisy) ===
def _cosine_alpha_bar(frac, s=0.008):
    # frac ∈ [0,1] 對應從 t=T 到 t=0 的「連續時間」比例
    return np.cos(( (frac + s) / (1.0 + s) ) * (np.pi/2.0))**2

import numpy as np

def _cosine_alpha_bar(frac, s=0.008):
    # frac in [0,1]
    t = (frac + s) / (1.0 + s) * (np.pi / 2.0)
    f = np.cos(t)**2
    ab = f / f[0]  # normalize so ab[0] = 1
    return np.clip(ab, 1e-8, 1.0)

class CosineNoiseForward:
    def __init__(self, steps=50, noise_scale=1.0, seed=None):
        self.steps = int(steps)
        self.noise_scale = float(noise_scale)  # 當旋鈕用；=1 才是理論方差
        self.seed = seed

    def _alpha_bar_seq(self):
        frac = np.linspace(0.0, 1.0, self.steps + 1, dtype=np.float32)
        return _cosine_alpha_bar(frac)

    def run_from_x0(self, x0):
        x0 = x0.astype(np.float32)
        ab = self._alpha_bar_seq()                 # shape: (steps+1,)
        beta = 1.0 - (ab[1:] / ab[:-1])            # shape: (steps,)
        beta = np.clip(beta, 1e-8, 0.999)          # 數值保險

        rng = np.random.default_rng(self.seed)
        xt = x0.copy()
        out = [xt.copy()]                          # out[0] = x0

        for b in beta:
            zt = rng.standard_normal(size=x0.shape).astype(np.float32)
            xt = np.sqrt(1.0 - b) * xt + np.sqrt(b) * (self.noise_scale * zt)
            out.append(xt.copy())
        return out  # 長度 steps+1，最後一張最吵

