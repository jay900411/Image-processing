
import os
import imageio.v3 as iio
import numpy as np
from skimage.transform import resize
from .utils import ensure_dir, to_uint8, minmax_normalize

class SyntheticDataset:
    """Generates a small set of grayscale images with edges/shapes and gradients."""
    def __init__(self, n=8, size=128, seed=42):
        self.n = n
        self.size = size
        self.rng = np.random.RandomState(seed)

    def generate(self):
        H = W = self.size
        out = []
        for i in range(self.n):
            x = np.zeros((H,W), dtype=np.float32)
            rr, cc = np.ogrid[:H, :W]
            # gradient background
            x += (rr / max(H-1,1)).astype(np.float32) * 0.5
            # circle
            cy, cx = H//2 + (i%3-1)*H//6, W//2 + ((i//3)%3-1)*W//6
            r = min(H,W)//6
            mask = (rr-cy)**2 + (cc-cx)**2 <= r*r
            x[mask] = 1.0
            out.append(x)
        return out

class FolderDataset:
    def __init__(self, directory, size=None):
        self.dir = directory
        self.size = size  # int or None
        self.paths = [os.path.join(directory, f) for f in os.listdir(directory)
                      if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        self.paths.sort()

    def generate(self):
        imgs = []
        for p in self.paths:
            im = iio.imread(p)
            if im.ndim == 3:
                im = im.mean(axis=2)  # gray
            im = minmax_normalize(im.astype(np.float32))
            if self.size and self.size > 0 and im.shape != (self.size, self.size):
                im = resize(im, (self.size, self.size), anti_aliasing=True).astype(np.float32)
            imgs.append(im)
        return imgs
