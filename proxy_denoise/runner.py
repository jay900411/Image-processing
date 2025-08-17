
import os, json
from .main import run_experiment

out_dir = "./outputs"
run_experiment(out_dir=out_dir,
               n_images=2,
               img_size=96,
               steps=100,
               mode="gaussian_then_bilateral",
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
               noise_scale=0.5 )

print("DONE. Outputs in:", out_dir)
