import os
import re
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 容錯：某些檔名可能少一個字母或大小寫不同
CANDIDATES = {
    "low_raw":      ["low_raw.png", "low_image.png", "low.png"],
    "high_raw":     ["high_raw.png", "high_image.png", "high.png"],
    "low_eq":       ["low_eq.png"],
    "high_eq":      ["high_eq.png"],
    "low_fft":      ["low_fourier.png", "low_fft.png"],
    "high_fft":     ["high_fourier.png", "high_fft.png"],
    "low_dct":      ["low_dct.png"],
    "high_dct":     ["high_dct.png"],
    "low_stft":     ["low_stft2d_avg.png", "low_stft2d_av.png", "low_stft.png"],
    "high_stft":    ["high_stft2d_avg.png", "high_stft2d_av.png", "high_stft.png"],
    "low_ridge":    ["low_ridgelet.png", "low_curvelet.png"],
    "high_ridge":   ["high_ridgelet.png", "high_curvelet.png"],
}

def find_first(path: Path, names):
    for n in names:
        p = path / n
        if p.exists():
            return p
    return None

def load_img(path: Path):
    if path is None or not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def save_grid(images, titles, outpath: Path, ncols=2, pad=10, bg="white"):
    """images: list of PIL images or None; titles 同長度，可為 ''；自動調整 rows/cols。"""
    imgs = images[:]
    # 先把缺圖補上灰格
    w = max((im.width for im in imgs if im), default=512)
    h = max((im.height for im in imgs if im), default=384)
    placeholder = Image.new("RGB", (w, h), (230,230,230))
    imgs = [im if im else placeholder for im in imgs]

    # 單純用 matplotlib 佈局
    n = len(imgs)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * 4
    fig_h = nrows * 3
    plt.figure(figsize=(fig_w, fig_h))
    for i, (im, title) in enumerate(zip(imgs, titles), start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(im)
        plt.axis("off")
        if title:
            plt.title(title, fontsize=10)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

def build_basic_panels(pair_dir: Path):
    panels_dir = pair_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    # 1) 空間域：raw vs eq
    imgs = [
        load_img(find_first(pair_dir, CANDIDATES["low_raw"])),
        load_img(find_first(pair_dir, CANDIDATES["high_raw"])),
        load_img(find_first(pair_dir, CANDIDATES["low_eq"])),
        load_img(find_first(pair_dir, CANDIDATES["high_eq"])),
    ]
    titles = ["Low (raw)","High (raw)","Low (CLAHE)","High (CLAHE)"]
    save_grid(imgs, titles, panels_dir / "compare_spatial.png", ncols=2)

    # 2) FFT
    imgs = [
        load_img(find_first(pair_dir, CANDIDATES["low_fft"])),
        load_img(find_first(pair_dir, CANDIDATES["high_fft"])),
        load_img(find_first(pair_dir, CANDIDATES["low_stft"])),
        load_img(find_first(pair_dir, CANDIDATES["high_stft"])),
    ]
    titles = ["Fourier (Low)","Fourier (High)", "STFT (Low)","STFT (High)"]
    save_grid(imgs, titles, panels_dir / "compare_fft_stft.png", ncols=2)

    # 3) DCT
    imgs = [
        load_img(find_first(pair_dir, CANDIDATES["low_dct"])),
        load_img(find_first(pair_dir, CANDIDATES["high_dct"])),
    ]
    titles = ["DCT (Low)","DCT (High)"]
    save_grid(imgs, titles, panels_dir / "compare_dct.png", ncols=2)

    # 5) Ridgelet / Curvelet
    imgs = [
        load_img(find_first(pair_dir, CANDIDATES["low_ridge"])),
        load_img(find_first(pair_dir, CANDIDATES["high_ridge"])),
    ]
    titles = ["Ridgelet (Low)","Ridgelet (High)"]
    save_grid(imgs, titles, panels_dir / "compare_ridgelet.png", ncols=2)

def build_wavelet_panels(pair_dir: Path):
    """依層出圖：每層一張面板，含 Low/High 的 LH/HL/HH 三個子帶（共 6 張圖）。"""
    wav_dir = pair_dir / "wavelet"
    if not wav_dir.exists():
        return
    # 找出最大層數
    patt = re.compile(r".*_(LH|HL|HH)_L(\d+)\.png$", re.IGNORECASE)
    levels = set()
    for p in wav_dir.glob("*.png"):
        m = patt.match(p.name)
        if m:
            levels.add(int(m.group(2)))
    if not levels:
        return

    panels_dir = pair_dir / "panels"
    for L in sorted(levels):
        # 對每層，取 low/high + 三個子帶
        triplets = [("LH", "Low"), ("HL", "Low"), ("HH", "Low"),
                    ("LH", "High"),("HL", "High"),("HH", "High")]
        imgs, titles = [], []
        for band, tag in triplets:
            fname = f"{('low' if tag=='Low' else 'high')}_wavelet_{band}_L{L}.png"
            # 舊命名相容：若你實際輸出是 "{tag}_wavelet_LH_L{L}.png"，也處理
            cand = [
                wav_dir / fname,
                wav_dir / f"{('low' if tag=='Low' else 'high')}_{band}_L{L}.png",
                wav_dir / f"{('low' if tag=='Low' else 'high')}_wavelet_{band.lower()}_L{L}.png",
                wav_dir / f"{('low' if tag=='Low' else 'high')}_wavelet_{band}_l{L}.png",
                wav_dir / f"{('low' if tag=='Low' else 'high')}_{band.lower()}_L{L}.png",
            ]
            chosen = next((c for c in cand if c.exists()), None)
            imgs.append(load_img(chosen))
            titles.append(f"{tag} {band} L{L}")
        save_grid(imgs, titles, panels_dir / f"compare_wavelet_L{L}.png", ncols=3)

def process_root(root: Path):
    count = 0
    for name in sorted(os.listdir(root)):
        p = root / name
        if p.is_dir():
            print(f"[PANEL] {p}")
            build_basic_panels(p)
            build_wavelet_panels(p)
            count += 1
    print(f"✅ 完成 {count} 個資料夾的面板輸出")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, help="results_batch 根目錄")
    ap.add_argument("--pair-dir", type=str, help="單一 pair 目錄")
    args = ap.parse_args()

    if args.pair_dir:
        d = Path(args.pair_dir)
        build_basic_panels(d)
        build_wavelet_panels(d)
        print("✅ 單一資料夾面板完成")
    elif args.root:
        process_root(Path(args.root))
    else:
        print("請指定 --root 或 --pair-dir")

if __name__ == "__main__":
    main()
