
# Proxy Denoise (OOP, no OpenCV)

This project mimics diffusion forward / reverse processes to produce **denoise trajectories** and **metrics**.  
It implements two forward (degradation) processes and uses their **reversed sequences** as approximate denoising trajectories.

## Processes
1. **CosineSmooth**: Cosine noise schedule with **spatially-smoothed Gaussian noise** (reduces grain).
2. **GaussianBilateralBlend**: Each step blends **Gaussian blur** and **bilateral filter** (always mixed, never switching).

## Outputs (per input image)
```
outputs/
  forward_noise/      imgXX_trajectory.png     # forward cosine-smooth grid
  forward_smooth/     imgXX_trajectory.png     # forward gaussian+bilateral grid
  denoise_noise/      imgXX_*.png              # reverse grid + per-metric curves
  denoise_smooth/     imgXX_*.png
  settings.json
  report.json
```
Per-metric files include:
- `H_vs_t` (gradient entropy)
- `Ehigh_vs_t`, `Elow_vs_t` (FFT high/low energy via circular mask)
- `MI_vs_t` and `MI_slope_vs_t` (mutual information between consecutive reverse steps)
- `R_vs_t` (mean gradient magnitude)
- `rho_vs_t` (edge density via Otsu-thresholded Sobel magnitude)

Each denoise folder also contains `imgXX_trajectory.png` – the **5×10** mosaic of the **reverse** sequence.

How to run
==========
1) 
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate



# Windows
python -m venv .venv
.\.venv\Scripts\activate

## Install
```bash
python -m pip install -U pip
python -m pip install -r expA_proxy_denoise2/requirements.txt
```

## Run
```bash
python -m expA_proxy_denoise2.runner
```
- Put your input images in `data/` (any PNG/JPG). Color images are converted to grayscale.
- `--img_size` resizes to a square for speed (default 256). Set to 0 to keep original size (bilateral may be slow).

## Early-stop heuristic
We detect a plateau on `MI_vs_t` using a moving slope threshold; suggested early-stop `t*` is written into `report.json`.
