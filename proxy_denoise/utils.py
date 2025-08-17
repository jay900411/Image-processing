
import os
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def minmax_normalize(img, eps=1e-8):
    mn, mx = img.min(), img.max()
    if mx - mn < eps:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn + eps)

def to_uint8(img):
    x = minmax_normalize(img) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)

def percentile_clip(x, lo=0.0, hi=100.0):
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if b <= a: 
        return x.copy()
    y = np.clip(x, a, b)
    return (y - a) / (b - a + 1e-8)

def meshgrid_2d(h, w):
    ys = np.arange(h).reshape(-1,1)
    xs = np.arange(w).reshape(1,-1)
    return ys, xs

def pad_reflect(img, r):
    # reflect padding for 2D arrays
    return np.pad(img, ((r,r),(r,r)), mode="reflect")
