
import os
import csv
import argparse
from typing import List, Tuple, Dict
from pathlib import Path

from oop_uav_quality_analysis import ImagePairAnalyzer, AnalysisConfig, STFTConfig, WaveletConfig

def read_pairs_from_csv(csv_path: str) -> List[Tuple[str, str, str]]:
    pairs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {"low", "high"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must include columns: 'low', 'high' (optional 'id').")
        for i, row in enumerate(reader):
            pid = row.get("id") or f"pair_{i:04d}"
            low = row["low"]
            high = row["high"]
            pairs.append((pid, low, high))
    return pairs

def match_pairs_by_basename(low_dir: str, high_dir: str) -> List[Tuple[str, str, str]]:
    low_dir = Path(low_dir)
    high_dir = Path(high_dir)
    if not low_dir.exists() or not high_dir.exists():
        raise FileNotFoundError("Both --low-dir and --high-dir must exist.")
    low_map: Dict[str, Path] = {p.stem: p for p in low_dir.iterdir() if p.is_file()}
    high_map: Dict[str, Path] = {p.stem: p for p in high_dir.iterdir() if p.is_file()}
    keys = sorted(set(low_map.keys()) & set(high_map.keys()))
    if not keys:
        raise ValueError("No matching filenames (by stem) between low-dir and high-dir.")
    pairs: List[Tuple[str, str, str]] = []
    for k in keys:
        pairs.append((k, str(low_map[k]), str(high_map[k])))
    return pairs

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run_batch(pairs: List[Tuple[str, str, str]], out_root: str,
              force_resize_high_to_low: bool = True,
              apply_clahe: bool = True,
              clahe_clip_limit: float = 0.01,
              stft_win: int = 64, stft_hop: int = 32,
              wavelet_name: str = "db2", wavelet_level: int = 2):
    ensure_dir(out_root)
    for pid, low, high in pairs:
        out_dir = os.path.join(out_root, pid)
        cfg = AnalysisConfig(
            force_resize_high_to_low=force_resize_high_to_low,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            stft=STFTConfig(win_size=stft_win, hop=stft_hop),
            wavelet=WaveletConfig(wavelet=wavelet_name, level=wavelet_level),
            out_dir=out_dir
        )
        print(f"[INFO] Analyzing pair '{pid}' -> {out_dir}")
        analyzer = ImagePairAnalyzer(cfg)
        try:
            analyzer.run(low, high)
            print(f"[DONE] {pid}: metrics summary written to {out_dir}")
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="Batch runner for UAV/MRI image pair analysis")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs-csv", type=str, help="CSV with columns low,high (optional 'id')")
    src.add_argument("--low-dir", type=str, help="Directory of low-quality images (match by basename)")
    ap.add_argument("--high-dir", type=str, help="Directory of high-quality images (required with --low-dir)")
    ap.add_argument("--out-root", type=str, default="results_batch", help="Output root directory")

    ap.add_argument("--no-resize", action="store_true", help="Do not resize high to match low")
    ap.add_argument("--no-clahe", action="store_true", help="Disable CLAHE before analysis")
    ap.add_argument("--clahe-clip", type=float, default=0.01, help="CLAHE clip limit")
    ap.add_argument("--stft-win", type=int, default=64, help="STFT window size")
    ap.add_argument("--stft-hop", type=int, default=32, help="STFT hop size")
    ap.add_argument("--w-name", type=str, default="db2", help="Wavelet name")
    ap.add_argument("--w-level", type=int, default=2, help="Wavelet levels")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.pairs_csv:
        pairs = read_pairs_from_csv(args.pairs_csv)
    else:
        if not args.high_dir:
            raise SystemExit("--high-dir is required when using --low-dir matching mode.")
        pairs = match_pairs_by_basename(args.low_dir, args.high_dir)

    run_batch(
        pairs=pairs,
        out_root=args.out_root,
        force_resize_high_to_low=(not args.no_resize),
        apply_clahe=(not args.no_clahe),
        clahe_clip_limit=args.clahe_clip,
        stft_win=args.stft_win,
        stft_hop=args.stft_hop,
        wavelet_name=args.w_name,
        wavelet_level=args.w_level
    )

if __name__ == "__main__":
    main()
