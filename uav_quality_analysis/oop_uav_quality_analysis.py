
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft as spfft
from skimage import io as skio
from skimage import color, transform, filters, feature, util, exposure
import pywt

@dataclass
class STFTConfig:
    win_size: int = 64
    hop: int = 32

@dataclass
class WaveletConfig:
    wavelet: str = "db2"
    level: int = 2

@dataclass
class AnalysisConfig:
    force_resize_high_to_low: bool = True
    apply_clahe: bool = True
    clahe_clip_limit: float = 0.01
    stft: STFTConfig = STFTConfig()
    wavelet: WaveletConfig = WaveletConfig()
    out_dir: str = "results"

class ImageUtils:
    @staticmethod
    def ensure_gray_float01(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = color.rgb2gray(img)
        img = util.img_as_float32(img)
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def resize_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if a.shape != b.shape:
            b_resized = transform.resize(
                b, a.shape, order=1, mode="reflect",
                anti_aliasing=True, preserve_range=True
            )
            b_resized = ImageUtils.ensure_gray_float01(b_resized)
            return a, b_resized
        return a, b

    @staticmethod
    def mkout(path: str) -> None:
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def to_display(arr):
        import numpy as np
        x = np.log1p(np.abs(arr).astype(np.float32))
        m = float(x.max()) if x.size else 0.0
        if m > 0:
            x = x / m
        return x    

    @staticmethod
    def save_img(im: np.ndarray, title: str, out_path: str, cmap: Optional[str] = None) -> None:
        plt.figure()
        if im.ndim == 2:
            plt.imshow(im, cmap=cmap or "gray")
        else:
            plt.imshow(im)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

class Transforms:
    def __init__(self, wavelet_cfg: WaveletConfig, stft_cfg: STFTConfig):
        self.wavelet_cfg = wavelet_cfg
        self.stft_cfg = stft_cfg

    @staticmethod
    def fourier_spectrum(img: np.ndarray) -> np.ndarray:
        F = spfft.fft2(img)
        F = spfft.fftshift(F)
        mag = np.abs(F)
        return np.log1p(mag)

    @staticmethod
    def dct2(img: np.ndarray) -> np.ndarray:
        return spfft.dctn(img, type=2, norm="ortho")

    def wavelet(self, img: np.ndarray):
        return pywt.wavedec2(img, wavelet=self.wavelet_cfg.wavelet, level=self.wavelet_cfg.level)

    def stft2d_avg_spectrum(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape
        win = np.hanning(self.stft_cfg.win_size)[:, None] * np.hanning(self.stft_cfg.win_size)[None, :]
        specs = []
        for y in range(0, H - self.stft_cfg.win_size + 1, self.stft_cfg.hop):
            for x in range(0, W - self.stft_cfg.win_size + 1, self.stft_cfg.hop):
                patch = img[y:y+self.stft_cfg.win_size, x:x+self.stft_cfg.win_size] * win
                F = spfft.fft2(patch)
                F = spfft.fftshift(F)
                specs.append(np.log1p(np.abs(F)))
        if not specs:
            return np.zeros((self.stft_cfg.win_size, self.stft_cfg.win_size), dtype=np.float32)
        return np.mean(np.stack(specs, axis=0), axis=0)

    @staticmethod
    def ridgelet_like(img: np.ndarray, thetas: Optional[np.ndarray] = None) -> np.ndarray:
        from skimage.transform import radon
        if thetas is None:
            thetas = np.linspace(0., 180., 181, endpoint=True)
        sino = radon(img.astype(np.float64), theta=thetas, circle=False)
        coeff = spfft.dct(sino, type=2, norm="ortho", axis=0)
        return np.log1p(np.abs(coeff))

class Metrics:
    @staticmethod
    def high_frequency_ratio(img: np.ndarray, cutoff_ratio: float = 0.25) -> float:
        H, W = img.shape
        F = spfft.fft2(img)
        F = spfft.fftshift(F)
        P = np.abs(F)**2
        cy, cx = H//2, W//2
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        r_max = min(H, W)/2.0
        r_cut = cutoff_ratio * r_max
        low = P[r <= r_cut].sum()
        allp = P.sum() + 1e-12
        high = allp - low
        return float(high / allp)

    @staticmethod
    def gradient_entropy(img: np.ndarray, nbins: int = 64) -> float:
        gx = filters.sobel_h(img)
        gy = filters.sobel_v(img)
        gm = np.hypot(gx, gy)
        hist, _ = np.histogram(gm.ravel(), bins=nbins, range=(0, gm.max() + 1e-12), density=True)
        hist = hist + 1e-12
        return float(-np.sum(hist * np.log2(hist)))

    @staticmethod
    def edge_density(img: np.ndarray, sigma: float = 1.0,
                     low_thresh: Optional[float] = None, high_thresh: Optional[float] = None) -> float:
        if low_thresh is None or high_thresh is None:
            edges = feature.canny(img, sigma=sigma)
        else:
            edges = feature.canny(img, sigma=sigma, low_threshold=low_thresh, high_threshold=high_thresh)
        return float(edges.mean())

    @staticmethod
    def orientation_variance(ridge_map: np.ndarray) -> float:
        return float(np.var(np.mean(ridge_map, axis=0)))

class Reporter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        ImageUtils.mkout(out_dir)
        ImageUtils.mkout(os.path.join(out_dir, "wavelet"))

    def save_wavelet_coeffs(self, coeffs, title_prefix: str):
        LL = coeffs[0]
        ImageUtils.save_img(np.abs(LL), f"{title_prefix} - LL", os.path.join(self.out_dir, "wavelet", f"{title_prefix}_LL.png"), cmap="magma")
        for i, (LH, HL, HH) in enumerate(coeffs[1:], start=1):
            ImageUtils.save_img(np.abs(LH), f"{title_prefix} - LH L{i}", os.path.join(self.out_dir, "wavelet", f"{title_prefix}_LH_L{i}.png"), cmap="magma")
            ImageUtils.save_img(np.abs(HL), f"{title_prefix} - HL L{i}", os.path.join(self.out_dir, "wavelet", f"{title_prefix}_HL_L{i}.png"), cmap="magma")
            ImageUtils.save_img(np.abs(HH), f"{title_prefix} - HH L{i}", os.path.join(self.out_dir, "wavelet", f"{title_prefix}_HH_L{i}.png"), cmap="magma")

    def save_core_visuals(self, tag: str, src_img: np.ndarray, spec: np.ndarray,
                          dct_abs: np.ndarray, stft_avg: np.ndarray, ridge_map: np.ndarray):
        ImageUtils.save_img(src_img, f"{tag.capitalize()}-quality (analysis image)", os.path.join(self.out_dir, f"{tag}_image.png"))
        ImageUtils.save_img(spec, f"Fourier log-magnitude ({tag})", os.path.join(self.out_dir, f"{tag}_fourier.png"), cmap="magma")
        ImageUtils.save_img(dct_abs, f"DCT magnitude ({tag})", os.path.join(self.out_dir, f"{tag}_dct.png"), cmap="magma")
        ImageUtils.save_img(stft_avg, f"2D STFT avg log-magnitude ({tag})", os.path.join(self.out_dir, f"{tag}_stft2d_avg.png"), cmap="magma")
        ImageUtils.save_img(ridge_map, f"Ridgelet (Radon + DCT) magnitude ({tag})", os.path.join(self.out_dir, f"{tag}_ridgelet.png"), cmap="magma")

    def save_raw_and_eq(self, low_raw, high_raw, low_eq, high_eq):
        if low_raw is not None:
            ImageUtils.save_img(low_raw, "Low-quality (raw color)", os.path.join(self.out_dir, "low_raw.png"))
        if high_raw is not None:
            ImageUtils.save_img(high_raw, "High-quality (raw color)", os.path.join(self.out_dir, "high_raw.png"))
        ImageUtils.save_img(low_eq, "Low-quality (CLAHE)", os.path.join(self.out_dir, "low_eq.png"))
        ImageUtils.save_img(high_eq, "High-quality (CLAHE)", os.path.join(self.out_dir, "high_eq.png"))

    def dump_metrics(self, metrics: Dict[str, Any]):
        with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def dump_summary(self, metrics: Dict[str, Any]):
        lines = []
        for k in ["high_freq_ratio_25", "high_freq_ratio_33", "gradient_entropy", "edge_density"]:
            v_low = metrics["low"][k]
            v_high = metrics["high"][k]
            trend = "↑" if v_high > v_low else "↓"
            lines.append(f"{k}: low={v_low:.4f}, high={v_high:.4f} ({trend})")
        lines.append(f"orientation_variance: low={metrics['orientation_variance']['low']:.4f}, high={metrics['orientation_variance']['high']:.4f}")
        with open(os.path.join(self.out_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines))

class ImagePairAnalyzer:
    def __init__(self, cfg: AnalysisConfig = AnalysisConfig()):
        self.cfg = cfg
        self.transforms = Transforms(cfg.wavelet, cfg.stft)
        self.reporter = Reporter(cfg.out_dir)

    def _prepare_inputs(self, path_low: str, path_high: str):
        low_raw = skio.imread(path_low)
        high_raw = skio.imread(path_high)

        low_gray = ImageUtils.ensure_gray_float01(low_raw)
        high_gray = ImageUtils.ensure_gray_float01(high_raw)

        if self.cfg.force_resize_high_to_low:
            high_gray = transform.resize(
                high_gray, low_gray.shape, order=1, mode="reflect",
                anti_aliasing=True, preserve_range=True
            )
            high_gray = ImageUtils.ensure_gray_float01(high_gray)

        if self.cfg.apply_clahe:
            low_eq = exposure.equalize_adapthist(low_gray, clip_limit=self.cfg.clahe_clip_limit)
            high_eq = exposure.equalize_adapthist(high_gray, clip_limit=self.cfg.clahe_clip_limit)
        else:
            low_eq, high_eq = low_gray, high_gray

        low_raw_color = low_raw if (low_raw.ndim == 3) else None
        high_raw_color = high_raw if (high_raw.ndim == 3) else None

        return (low_raw_color, high_raw_color, low_eq, high_eq)

    def _analyze_single(self, img: np.ndarray, tag: str) -> Dict[str, Any]:
        spec = self.transforms.fourier_spectrum(img)
        dct_coeff = self.transforms.dct2(img)
        dct_disp  = ImageUtils.to_display(dct_coeff)
        w = self.transforms.wavelet(img)
        stft_avg = self.transforms.stft2d_avg_spectrum(img)
        ridge_map = self.transforms.ridgelet_like(img)

        self.reporter.save_core_visuals(tag, img, spec, dct_disp, stft_avg, ridge_map)
        self.reporter.save_wavelet_coeffs(w, f"{tag}_wavelet")

        m = {
            "high_freq_ratio_25": Metrics.high_frequency_ratio(img, 0.25),
            "high_freq_ratio_33": Metrics.high_frequency_ratio(img, 1/3),
            "gradient_entropy": Metrics.gradient_entropy(img),
            "edge_density": Metrics.edge_density(img, sigma=1.0),
            "orientation_variance_local": Metrics.orientation_variance(ridge_map),
        }
        return {"visuals": {"fourier": spec.shape, "dct": dct_disp.shape, "stft": stft_avg.shape, "ridge": ridge_map.shape},
                "metrics": m}

    def run(self, path_low: str, path_high: str) -> Dict[str, Any]:
        ImageUtils.mkout(self.cfg.out_dir)
        low_raw, high_raw, low_eq, high_eq = self._prepare_inputs(path_low, path_high)
        self.reporter.save_raw_and_eq(low_raw, high_raw, low_eq, high_eq)

        res_low = self._analyze_single(low_eq, "low")
        res_high = self._analyze_single(high_eq, "high")

        metrics = {
            "low": res_low["metrics"],
            "high": res_high["metrics"],
            "orientation_variance": {
                "low": res_low["metrics"]["orientation_variance_local"],
                "high": res_high["metrics"]["orientation_variance_local"],
            },
            "config": asdict(self.cfg),
        }

        self.reporter.dump_metrics(metrics)
        self.reporter.dump_summary(metrics)
        return metrics

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="OOP UAV/MRI Image Quality Transform Analysis")
    p.add_argument("--low", required=True, help="Path to low-quality image (low-light + low-res)")
    p.add_argument("--high", required=True, help="Path to high-quality image (high-light + high-res)")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--no-resize", action="store_true", help="Do not resize high to match low")
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE before analysis")
    p.add_argument("--stft-win", type=int, default=64, help="STFT window size")
    p.add_argument("--stft-hop", type=int, default=32, help="STFT hop size")
    p.add_argument("--w-name", type=str, default="db2", help="Wavelet name")
    p.add_argument("--w-level", type=int, default=2, help="Wavelet levels")
    return p.parse_args()

def main():
    args = _parse_args()
    cfg = AnalysisConfig(
        force_resize_high_to_low=(not args.no_resize),
        apply_clahe=(not args.no_clahe),
        stft=STFTConfig(win_size=args.stft_win, hop=args.stft_hop),
        wavelet=WaveletConfig(wavelet=args.w_name, level=args.w_level),
        out_dir=args.out
    )
    analyzer = ImagePairAnalyzer(cfg)
    analyzer.run(args.low, args.high)

if __name__ == "__main__":
    pass
