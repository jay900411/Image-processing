
import os
import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir, to_uint8, minmax_normalize
from .metrics import (
    gradient_magnitude, entropy_from_hist, edge_density,
    tiny_blur_for_metric, mutual_information_hist, fft_energy_banded
)

class EarlyStopper:
    def __init__(self, ref_mu_H, ref_std_H, ref_mu_R, ref_std_R, alpha=1.0, beta=1.0, plateau_K=3, plateau_eps=1e-3):
        self.ref_mu_H = ref_mu_H
        self.ref_std_H = ref_std_H
        self.ref_mu_R = ref_mu_R
        self.ref_std_R = ref_std_R
        self.alpha = alpha
        self.beta = beta
        self.K = plateau_K
        self.eps = plateau_eps

    def in_band(self, H, R):
        condH = abs(H - self.ref_mu_H) <= self.alpha * max(self.ref_std_H, 1e-6)
        condR = abs(R - self.ref_mu_R) <= self.beta * max(self.ref_std_R, 1e-6)
        return bool(condH and condR)

    def plateau(self, seq, t):
        if t < self.K: return False
        s = np.asarray(seq, dtype=np.float32)
        slopes = s[1:] - s[:-1]
        win = slopes[t-self.K:t]
        return bool(np.all(win <= self.eps))

    def decide(self, H_seq, R_seq):
        # simple heuristic: first t that is in band and MI slope plateaus would be better,
        # but here we just use in-band earliest t
        for t, (H, R) in enumerate(zip(H_seq, R_seq)):
            if self.in_band(H, R):
                return t
        return len(H_seq) - 1

class MetricComputer:
    def __init__(self, bins_entropy=256, bins_mi=64, edge_method="otsu", edge_thresh=85):
        self.bins_entropy = bins_entropy
        self.bins_mi = bins_mi
        self.edge_method = edge_method
        self.edge_thresh = edge_thresh

    def compute_all(self, x_clean, xt_seq):
        Hs, rhos, Elows, Ehighs, Ratios, MIs = [], [], [], [], [], []
        prev = None
        for xt in xt_seq:
            xts = tiny_blur_for_metric(xt, sigma=0.5)
            mag = gradient_magnitude(xts)
            mag_n = (np.log1p(mag * 3.0) - 0.0)  # already >=0
            mag_n = (mag_n - mag_n.min()) / (mag_n.max()-mag_n.min()+1e-8)
            H = entropy_from_hist(mag_n, bins=self.bins_entropy, clip=(1,99))
            rho = edge_density(xts, method=self.edge_method, thresh=self.edge_thresh)
            Elow, Ehigh, ratio = fft_energy_banded(xts, r_low=0.25, r_edge=0.65, r_notch=0.80)
            Hs.append(H); rhos.append(rho); Elows.append(Elow); Ehighs.append(Ehigh); Ratios.append(ratio)
            if prev is None:
                MIs.append(0.0)
            else:
                MIs.append(mutual_information_hist(prev, xts, bins=self.bins_mi))
            prev = xts
        return {"H": Hs, "rho": rhos, "Elow": Elows, "Ehigh": Ehighs, "R": Ratios, "MI": MIs}

class Plotter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        ensure_dir(out_dir)

    def _resolve(self, path):
        if os.path.isabs(path): return path
        return os.path.join(self.out_dir, path)

    def _plot_curve(self, ys, title, fname):
        full = self._resolve(fname)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        plt.figure(figsize=(6,4))
        plt.plot(ys)
        plt.title(title); plt.xlabel("t (step)");
        plt.tight_layout(); plt.savefig(full, dpi=160); plt.close()

    def curves(self, metrics_dict, prefix=""):
        for key in ["H","rho","Elow","Ehigh","R"]:
            self._plot_curve(metrics_dict[key], f"{prefix}{key} vs t", f"{prefix}{key}_vs_t.png")
        if len(metrics_dict.get("MI", []))>0:
            self._plot_curve(metrics_dict["MI"], f"{prefix}MI vs t", f"{prefix}MI_vs_t.png")    

    def mi_slope(self, mi_values, prefix=""):
        if len(mi_values) < 2: return
        slopes = (np.asarray(mi_values)[1:] - np.asarray(mi_values)[:-1]).tolist()
        self._plot_curve(slopes, f"{prefix}MI_slope vs t", f"{prefix}MI_slope_vs_t.png")    

    def save_trajectory_grid(self, xt_seq, fname, cols=10):  # default 10x10 for 100 steps
        full = self._resolve(fname)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        n = len(xt_seq)
        rows = int(np.ceil(n/cols))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
        axes = np.atleast_2d(axes).reshape(rows, cols)
        for i in range(rows*cols):
            ax = axes[i//cols, i%cols]
            ax.axis("off")
            if i < n:
                ax.imshow(xt_seq[i], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"t={i}", fontsize=8)
        plt.tight_layout()
        plt.savefig(full, dpi=160)
        plt.close()

def compute_reference_band(clean_imgs, metric_computer):
    Hs, Rs = [], []
    for x in clean_imgs:
        d = metric_computer.compute_all(x, [x])
        Hs.append(d["H"][0]); Rs.append(d["R"][0])
    muH, stdH = float(np.mean(Hs)), float(np.std(Hs)+1e-8)
    muR, stdR = float(np.mean(Rs)), float(np.std(Rs)+1e-8)
    return (muH, stdH, muR, stdR)
