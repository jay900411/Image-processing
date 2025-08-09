import os
import argparse
from pathlib import Path

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def fetch_visdrone_clean(clean_dir: str, num: int = 80, split: str = "train"):
    import deeplake
    import imageio.v2 as imageio
    import numpy as np

    hub = {
        "train": "hub://activeloop/visdrone-det-train",
        "val":   "hub://activeloop/visdrone-det-val",
    }
    ds_uri = hub.get(split, hub["train"])

    print(f"[INFO] Loading Deep Lake dataset (v3 API): {ds_uri}")
    # v3 用 load 沒問題；加 read_only 避免權限提示
    ds = deeplake.load(ds_uri, read_only=True)

    ensure_dir(clean_dir)
    saved = 0

    # v3：ds.tensors 是「屬性」(dict-like)，不要寫 ds.tensors()
    tensor = None
    tensor_dict = ds.tensors  # <-- 關鍵：這是屬性
    # 先找叫 'images' 的 tensor，找不到再 fallback
    if isinstance(tensor_dict, dict) and "images" in tensor_dict:
        tensor = tensor_dict["images"]
    else:
        # 任選第一個能 .numpy() 出來的 tensor
        for name, t in (tensor_dict.items() if isinstance(tensor_dict, dict) else []):
            try:
                _arr = t[0].numpy()
                if isinstance(_arr, np.ndarray):
                    tensor = t
                    break
            except Exception:
                continue

    if tensor is None:
        raise RuntimeError("Could not find an image tensor in the dataset (v3).")

    total = len(ds)
    print(f"[INFO] Dataset has {total} samples; will try to save {num}.")

    i = 0
    idx = 0
    while i < num and idx < total:
        try:
            arr = tensor[idx].numpy()  # v3 仍可用 t[i].numpy()
            # 常見格式：CHW -> 轉 HWC
            if arr.ndim == 3 and arr.shape[0] < 8:
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != "uint8":
                arr = arr.astype("uint8")

            out_path = Path(clean_dir) / f"uav_{idx:05d}.jpg"
            imageio.imwrite(out_path, arr)
            i += 1
            if i % 10 == 0:
                print(f"[INFO] Saved {i} images to {clean_dir} ...")
        except Exception as e:
            print(f"[WARN] Skipping index {idx}: {e}")
        idx += 1

    print(f"[DONE] Saved {i} images to {clean_dir}")

def synthesize_pairs(clean_dir: str, out_low: str, out_high: str, pairs_csv: str,
                     downscale: float, blur: float, L_scale: float,
                     motion_len: int = 0, motion_angle: float = 0.0,
                     add_noise: bool = False, noise_sigma: float = 0.01,
                     interp: int = 1):
    try:
        from synthesize_pairs import make_pairs
        make_pairs(clean_dir, out_low, out_high, pairs_csv,
                   downscale, blur, L_scale,
                   motion_len, motion_angle,
                   add_noise, noise_sigma, interp)
    except ImportError:
        import subprocess, sys
        cmd = [
            sys.executable, "synthesize_pairs.py",
            "--clean-dir", clean_dir,
            "--out-low", out_low,
            "--out-high", out_high,
            "--pairs-csv", pairs_csv,
            "--downscale", str(downscale),
            "--blur", str(blur),
            "--L-scale", str(L_scale),
            "--motion-len", str(motion_len),
            "--motion-angle", str(motion_angle),
            "--interp", str(interp),
        ]
        if add_noise:
            cmd += ["--add-noise", "--noise-sigma", str(noise_sigma)]
        print("[INFO] Running:", " ".join(cmd))
        subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser(description="Fetch clean VisDrone images and synthesize (low, high) pairs")
    ap.add_argument("--num", type=int, default=80, help="Number of clean images to fetch (50~100 recommended)")
    ap.add_argument("--split", type=str, default="train", choices=["train","val"], help="VisDrone split to use")
    ap.add_argument("--out-root", type=str, default="data", help="Root folder for outputs")
    ap.add_argument("--downscale", type=float, default=0.5, help="Downscale factor (0<d<1)")
    ap.add_argument("--blur", type=float, default=1.2, help="Gaussian blur sigma")
    ap.add_argument("--L-scale", type=float, default=0.7, help="Lab L channel scale (<1 darken)")
    ap.add_argument("--motion-len", type=int, default=0, help="Motion blur kernel length (0=off)")
    ap.add_argument("--motion-angle", type=float, default=0.0, help="Motion blur angle")
    ap.add_argument("--add-noise", action="store_true", help="Add Gaussian noise")
    ap.add_argument("--noise-sigma", type=float, default=0.01, help="Noise sigma")
    ap.add_argument("--interp", type=int, default=1, help="Resize interpolation order (1=bilinear,3=bicubic)")
    args = ap.parse_args()

    clean_dir = os.path.join(args.out_root, "clean_uav")
    out_low = os.path.join(args.out_root, "low")
    out_high = os.path.join(args.out_root, "high")
    pairs_csv = os.path.join(args.out_root, "pairs.csv")

    fetch_visdrone_clean(clean_dir, num=args.num, split=args.split)

    synthesize_pairs(clean_dir, out_low, out_high, pairs_csv,
                     args.downscale, args.blur, args.L_scale,
                     args.motion_len, args.motion_angle,
                     args.add_noise, args.noise_sigma,
                     args.interp)

    print(f"""
[OK] Finished.
Clean images:   {clean_dir}
Low-quality:    {out_low}
High-quality:   {out_high}
Manifest CSV:   {pairs_csv}
Next:           python main_batch.py --pairs-csv {pairs_csv} --out-root results_batch
""")

if __name__ == "__main__":
    main()
