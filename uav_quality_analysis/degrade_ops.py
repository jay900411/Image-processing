
import numpy as np
from dataclasses import dataclass
from skimage import color, transform, util
from scipy.ndimage import gaussian_filter, convolve

@dataclass
class DegradeConfig:
    downscale: float = 0.5
    interpolation_order: int = 1
    blur_sigma: float = 1.2
    motion_length: int = 0
    motion_angle_deg: float = 0.0
    L_scale: float = 0.7
    add_gaussian_noise: bool = False
    noise_sigma: float = 0.01

def _to_float01(img: np.ndarray) -> np.ndarray:
    img = util.img_as_float32(img)
    return np.clip(img, 0.0, 1.0)

def _apply_lab_L_scale(img_rgb01: np.ndarray, L_scale: float) -> np.ndarray:
    lab = color.rgb2lab(img_rgb01)
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    L = np.clip(L * L_scale, 0, 100)
    lab2 = np.stack([L, a, b], axis=-1)
    out = color.lab2rgb(lab2)
    return np.clip(out, 0.0, 1.0)

def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)
    k = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    for t in np.linspace(-center, center, length):
        x = int(round(center + t*dx))
        y = int(round(center + t*dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    if s > 0:
        k /= s
    else:
        k[center, center] = 1.0
    return k

def degrade_image(clean_img: np.ndarray, cfg: DegradeConfig) -> np.ndarray:
    x = _to_float01(clean_img)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    H, W, _ = x.shape

    if 0 < cfg.downscale < 1.0:
        h2, w2 = max(2, int(H * cfg.downscale)), max(2, int(W * cfg.downscale))
        small = transform.resize(x, (h2, w2), order=cfg.interpolation_order, mode='reflect', anti_aliasing=True, preserve_range=True)
        x = transform.resize(small, (H, W), order=cfg.interpolation_order, mode='reflect', anti_aliasing=True, preserve_range=True)
        x = _to_float01(x)

    if cfg.blur_sigma > 0:
        x = gaussian_filter(x, sigma=(cfg.blur_sigma, cfg.blur_sigma, 0))

    if cfg.motion_length and cfg.motion_length > 1:
        k = _motion_kernel(cfg.motion_length, cfg.motion_angle_deg)
        x = np.stack([convolve(x[:,:,c], k, mode='reflect') for c in range(x.shape[2])], axis=-1)

    x = _apply_lab_L_scale(x, cfg.L_scale)

    if cfg.add_gaussian_noise and cfg.noise_sigma > 0:
        noise = np.random.normal(0.0, cfg.noise_sigma, size=x.shape).astype(np.float32)
        x = np.clip(x + noise, 0.0, 1.0)

    return x
