
import numpy as np
from skimage.filters import sobel, threshold_otsu
from scipy.ndimage import gaussian_filter

def gradient_magnitude(img: np.ndarray):
    """Use skimage.sobel on [0,1] float image."""
    x = img.astype(np.float32)
    return np.abs(sobel(x))

def entropy_from_hist(values_img: np.ndarray, bins=256, clip=(1,99)):
    """Histogram entropy on normalized values (auto percentile clip)."""
    v = values_img.astype(np.float32)
    lo, hi = np.percentile(v, clip)
    if hi <= lo:
        v = (v - v.min()) / (v.max()-v.min()+1e-8)
    else:
        v = np.clip(v, lo, hi)
        v = (v - lo) / (hi - lo + 1e-8)
    hist, _ = np.histogram(v, bins=bins, range=(0,1), density=True)
    hist = hist + 1e-12
    return float(-(hist * np.log(hist)).sum())

def edge_density(img: np.ndarray, method="otsu", thresh=85):
    g = gradient_magnitude(img)
    if method == "otsu":
        t = threshold_otsu(g)
    else:
        t = float(thresh) / 255.0
    edges = (g >= t).astype(np.float32)
    return float(edges.mean())

def tiny_blur_for_metric(img: np.ndarray, sigma=0.5):
    return gaussian_filter(img.astype(np.float32), sigma=sigma)

def mutual_information_hist(x: np.ndarray, y: np.ndarray, bins=64, eps=1e-12):
    """MI between two images in [0,1]. Robust uint8 histogram."""
    a = (np.clip(x,0,1)*255).astype(np.uint8).ravel().astype(np.int32)
    b = (np.clip(y,0,1)*255).astype(np.uint8).ravel().astype(np.int32)
    H, _, _ = np.histogram2d(a, b, bins=bins, range=[[0,255],[0,255]])
    Pxy = H / max(H.sum(), 1.0)
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = Pxy / (Px @ Py + eps)
        frac[~np.isfinite(frac)] = 1.0
        logf = np.log(frac + eps)
    I = (Pxy * logf).sum()
    return float(I)

def fft_energy_banded(img: np.ndarray, r_low: float=0.25, r_edge: float=0.65, r_notch: float=0.80, window: bool=True):
    """Return (Elow, Eband, ratio) with ultra-high freq ignored beyond r_notch."""
    I = img.astype(np.float32)
    if window:
        wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(I.shape[0]) / max(1, I.shape[0]-1))
        wx = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(I.shape[1]) / max(1, I.shape[1]-1))
        I = I * wy[:, None] * wx[None, :]
    F = np.fft.fftshift(np.fft.fft2(I))
    P = (F.real**2 + F.imag**2).astype(np.float64)
    H, W = I.shape
    cy, cx = (H-1)/2.0, (W-1)/2.0
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    r_max = np.sqrt(cy**2 + cx**2) + 1e-8
    r_norm = r / r_max

    Elow = float(P[r_norm < r_low].sum())
    Eband = float(P[(r_norm >= r_low) & (r_norm <= r_edge)].sum())
    # ignore r > r_notch
    denom = Elow + Eband + 1e-8
    ratio = float(Eband / denom)
    return Elow, Eband, ratio
