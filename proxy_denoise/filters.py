
import numpy as np
from .utils import pad_reflect

class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def _kernel_1d(self, sigma, radius=None):
        if sigma <= 0: 
            return np.array([1.0])
        if radius is None:
            radius = int(np.ceil(3*sigma))
        xs = np.arange(-radius, radius+1)
        k = np.exp(-(xs**2)/(2*sigma*sigma))
        k = k / k.sum()
        return k

    def apply(self, img):
        # separable conv
        k = self._kernel_1d(self.sigma)
        r = (len(k)-1)//2
        # horizontal
        pad = pad_reflect(img, r)
        tmp = np.zeros_like(img)
        for y in range(img.shape[0]):
            row = pad[y+r, :]
            # 1D conv across x
            for x in range(img.shape[1]):
                window = row[x:x+2*r+1]
                tmp[y, x] = (window * k).sum()
        # vertical
        pad2 = pad_reflect(tmp, r)
        out = np.zeros_like(img)
        for x in range(img.shape[1]):
            col = pad2[:, x+r]
            for y in range(img.shape[0]):
                window = col[y:y+2*r+1]
                out[y, x] = (window * k).sum()
        return np.clip(out, 0.0, 1.0)

class BilateralFilter:
    """
    Simple bilateral filter (slow but dependency-free).
    """
    def __init__(self, sigma_s=2.0, sigma_r=0.1, radius=None):
        self.sigma_s = float(sigma_s)
        self.sigma_r = float(sigma_r)
        if radius is None:
            radius = int(np.ceil(2*sigma_s))
        self.radius = int(radius)

    def apply(self, img):
        r = self.radius
        h, w = img.shape
        pad = pad_reflect(img, r)
        ys = np.arange(-r, r+1).reshape(-1,1)
        xs = np.arange(-r, r+1).reshape(1,-1)
        spatial = np.exp(-(ys**2 + xs**2)/(2*self.sigma_s*self.sigma_s))
        out = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                patch = pad[y:y+2*r+1, x:x+2*r+1]
                center = pad[y+r, x+r]
                range_w = np.exp(-((patch - center)**2)/(2*self.sigma_r*self.sigma_r))
                wgt = spatial * range_w
                wgt_sum = wgt.sum()
                if wgt_sum <= 1e-12:
                    out[y,x] = center
                else:
                    out[y,x] = (wgt * patch).sum() / wgt_sum
        return np.clip(out, 0.0, 1.0)

def GaussianBilateral(img, steps=50, mode="hybrid", **kwargs):
    """
    高斯+雙邊濾波模擬 forward 加噪，使用線性 sigma 遞增。
    """
    
    return forward_smooth_hybrid(img, steps, **kwargs)

import numpy as np

def forward_smooth_hybrid(
    x0: np.ndarray,
    steps: int,
    *,
    # 兩個濾波器的混合權重：w_g * Gaussian + (1 - w_g) * Bilateral
    w_g_start: float = 0.5,
    w_g_end: float   = 0.5,

    # Gaussian 參數（隨步數單調變化，避免突變）
    sigma_g_start: float = 0.6,
    sigma_g_end: float   = 2.4,

    # Bilateral 參數（空間尺度 ↑、range尺度 ↓，越走越糊但仍保邊）
    sigma_s_start: float = 1.2,
    sigma_s_end: float   = 2.4,
    sigma_r_start: float = 0.12,   # 像素值範圍正規化到[0,1]時的建議量級
    sigma_r_end: float   = 0.08,

    # 每步更新步長（越後面可稍小，避免過衝）
    k_start: float = 0.30,
    k_end: float   = 0.15,

    # 雜項
    radius_factor: float = 3.0,   # bilateral 半徑約 = radius_factor * sigma_s
):
    """
    從 x0 出發生成「越來越平滑」的前向序列（長度 steps+1）：
        x_{i+1} = (1 - k_i) * x_i + k_i * [ w_i * G(x_i) + (1 - w_i) * B(x_i) ]
    - 全程同時使用 Gaussian 與 Bilateral（無切換斷點）。
    - 參數採線性 schedule，確保連續。
    回傳: [x0, x1, ..., x_steps]
    """
    H, W = x0.shape
    xs = [x0.astype(np.float32).clip(0, 1)]

    wgs   = np.linspace(float(w_g_start), float(w_g_end), steps, dtype=np.float32)
    sig_g = np.linspace(float(sigma_g_start), float(sigma_g_end), steps, dtype=np.float32)
    sig_s = np.linspace(float(sigma_s_start), float(sigma_s_end), steps, dtype=np.float32)
    sig_r = np.linspace(float(sigma_r_start), float(sigma_r_end), steps, dtype=np.float32)
    ks    = np.linspace(float(k_start), float(k_end), steps, dtype=np.float32)

    for i in range(steps):
        xi = xs[-1]
        # 建立濾波器
        gb = GaussianBlur(float(max(1e-6, sig_g[i])))
        rad = int(np.ceil(radius_factor * float(sig_s[i])))
        rad = int(max(1, rad))
        bf = BilateralFilter(
            sigma_s=float(max(1e-6, sig_s[i])),
            sigma_r=float(max(1e-6, sig_r[i])),
            radius=rad
        )
        # 個別濾波
        xg = gb.apply(xi)
        xb = bf.apply(xi)
        # 加權混合
        mix = float(wgs[i]) * xg + (1.0 - float(wgs[i])) * xb
        # 小步長更新（避免一次跳太多）
        x_next = (1.0 - float(ks[i])) * xi + float(ks[i]) * mix
        xs.append(np.clip(x_next, 0.0, 1.0).astype(np.float32))

    return xs


