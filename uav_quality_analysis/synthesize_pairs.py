
import os
import argparse
from pathlib import Path
import csv

import imageio.v2 as imageio

from degrade_ops import DegradeConfig, degrade_image

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_image(p: Path):
    return p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def make_pairs(clean_dir: str, out_low: str, out_high: str, pairs_csv: str,
               downscale: float, blur: float, L_scale: float,
               motion_len: int = 0, motion_angle: float = 0.0,
               add_noise: bool = False, noise_sigma: float = 0.01,
               interpolation_order: int = 1):
    ensure_dir(out_low); ensure_dir(out_high)
    rows = []
    cfg = DegradeConfig(
        downscale=downscale,
        interpolation_order=interpolation_order,
        blur_sigma=blur,
        motion_length=motion_len,
        motion_angle_deg=motion_angle,
        L_scale=L_scale,
        add_gaussian_noise=add_noise,
        noise_sigma=noise_sigma
    )

    clean_dir = Path(clean_dir)
    imgs = sorted([p for p in clean_dir.iterdir() if p.is_file() and is_image(p)])
    if not imgs:
        raise SystemExit(f"No images found in {clean_dir}")

    for i, p in enumerate(imgs):
        clean = imageio.imread(p)
        degraded = degrade_image(clean, cfg)

        pid = p.stem
        low_path = Path(out_low) / f"{pid}.png"
        high_path = Path(out_high) / f"{pid}.png"

        imageio.imwrite(low_path, (degraded * 255).astype('uint8'))
        imageio.imwrite(high_path, clean)

        rows.append({"id": pid, "low": str(low_path), "high": str(high_path)})

    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "low", "high"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} pairs to {pairs_csv}")

def parse_args():
    ap = argparse.ArgumentParser(description="Synthesize (low, high) pairs from clean images")
    ap.add_argument("--clean-dir", required=True, help="Folder of clean images")
    ap.add_argument("--out-low", required=True, help="Output folder for degraded images")
    ap.add_argument("--out-high", required=True, help="Output folder for clean copies")
    ap.add_argument("--pairs-csv", required=True, help="CSV manifest to write")
    ap.add_argument("--downscale", type=float, default=0.5, help="Downscale factor (0<d<1)")
    ap.add_argument("--blur", type=float, default=1.2, help="Gaussian blur sigma")
    ap.add_argument("--L-scale", type=float, default=0.7, help="Lab L channel scale (<1 darken)")
    ap.add_argument("--motion-len", type=int, default=0, help="Motion blur kernel length (0=off)")
    ap.add_argument("--motion-angle", type=float, default=0.0, help="Motion blur angle in degrees")
    ap.add_argument("--add-noise", action="store_true", help="Add Gaussian noise")
    ap.add_argument("--noise-sigma", type=float, default=0.01, help="Noise sigma in [0,1] scale")
    ap.add_argument("--interp", type=int, default=1, help="Resize interpolation order (1=bilinear,3=bicubic)")
    return ap.parse_args()

def main():
    args = parse_args()
    make_pairs(
        clean_dir=args.clean_dir,
        out_low=args.out_low,
        out_high=args.out_high,
        pairs_csv=args.pairs_csv,
        downscale=args.downscale,
        blur=args.blur,
        L_scale=args.L_scale,
        motion_len=args.motion_len,
        motion_angle=args.motion_angle,
        add_noise=args.add_noise,
        noise_sigma=args.noise_sigma,
        interpolation_order=args.interp
    )

if __name__ == "__main__":
    main()
