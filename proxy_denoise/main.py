
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .utils import ensure_dir
from .dataset import FolderDataset
from .trajectory import CosineNoiseForward
from .experiment import MetricComputer, Plotter, EarlyStopper, compute_reference_band
from .filters import GaussianBilateral

def run_experiment(out_dir,
                   n_images=6,
                   img_size=96,
                   steps=100,
                   mode="hybrid",
                   edge_method="otsu",
                   edge_thresh=85,
                   rc_ratio=0.35,
                   bins_entropy=256,
                   bins_mi=64,
                   early_alpha=1.2,
                   early_beta=1.2,
                   plateau_K=4,
                   plateau_eps=2e-3,
                   seed=7,
                   noise_scale=0.5):

    ensure_dir(out_dir)
    plotter = Plotter(out_dir)
    metric_comp = MetricComputer(bins_entropy=bins_entropy, bins_mi=bins_mi,
                                 edge_method=edge_method, edge_thresh=edge_thresh)

    # === Load images ===
    data_dir = "/Users/chi.lai/Desktop/expA_proxy_denoise/data"
    ds = FolderDataset(directory=data_dir, size=img_size)
    imgs = ds.generate()
    if n_images is not None:
        imgs = imgs[:n_images]

    # === reference band ===
    muH, stdH, muR, stdR = compute_reference_band([x for x in imgs], metric_comp)
    stopper = EarlyStopper(muH, stdH, muR, stdR, alpha=early_alpha, beta=early_beta, plateau_K=plateau_K, plateau_eps=plateau_eps)

    # === per image ===
    reports = []
    for idx, x0 in enumerate(imgs):
        prefix_A = f"img{idx:02d}_"

        # A) Cosine noise forward
        fwd = CosineNoiseForward(steps=steps, seed=seed, noise_scale=noise_scale).run_from_x0(x0)

        denoise_seq = list(reversed(fwd))[:steps]  # t=1..steps
        # save reverse grid
        plotter.save_trajectory_grid(denoise_seq, fname=os.path.join("denoise_noise", prefix_A + "trajectory.png"), cols=10)
        metrics_A = metric_comp.compute_all(x0, denoise_seq)
        t_star_A = stopper.decide(metrics_A["H"], metrics_A["R"])
        plotter.curves(metrics_A,   prefix=os.path.join("denoise_noise",  prefix_A))
        plotter.mi_slope(metrics_A.get("MI", []), prefix=os.path.join("denoise_noise",  prefix_A))

        # B) Gaussian + Bilateral hybrid forward
        seq_fwd_smooth = GaussianBilateral(x0, steps=steps, mode=mode)
        seq_den_smooth = list(reversed(seq_fwd_smooth))[:steps]
        plotter.save_trajectory_grid(seq_den_smooth, fname=os.path.join("denoise_smooth", prefix_A + "trajectory.png"), cols=10)
        metrics_B = metric_comp.compute_all(x0, seq_den_smooth)
        t_star_B = stopper.decide(metrics_B["H"], metrics_B["R"])
        plotter.curves(metrics_B,   prefix=os.path.join("denoise_smooth",  prefix_A))
        plotter.mi_slope(metrics_B.get("MI", []), prefix=os.path.join("denoise_smooth",  prefix_A))

        reports.append({"index": idx, "t_early_stop_denoise_noise": int(t_star_A), "t_early_stop_denoise_smooth": int(t_star_B)})

    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump({
            "img_size": img_size, "steps": steps, "mode": mode,
            "edge_method": edge_method, "edge_thresh": edge_thresh,
            "rc_ratio": rc_ratio, "bins_entropy": bins_entropy, "bins_mi": bins_mi,
            "early_alpha": early_alpha, "early_beta": early_beta,
            "plateau_K": plateau_K, "plateau_eps": plateau_eps,
            "seed": seed, "noise_scale": noise_scale
        }, f, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(reports, f, indent=2)

if __name__ == "__main__":
    out = "./outputs"
    run_experiment(out_dir=out)
