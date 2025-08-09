# Image-processing
Some implementations

### `uav_quality_analysis`

#### Comparing high and low resolution frequency differences under various conversions.

This project compares low-light + low-resolution (simulated) UAV images with high-quality images using various frequency and spatial domain transforms, producing visual and quantitative results.

#### Key Features
##### -Data Preparation

-Download clean UAV images from the VisDrone dataset

-Simulate degraded versions via downscaling, Gaussian blur, and brightness reduction (LAB L-channel scaling)

-Generate pairs.csv mapping high–low image pairs

##### -Transform & Analysis

-Spatial: original, histogram equalized images

-Frequency: Fourier Transform (FFT), Discrete Cosine Transform (DCT), Short-Time Fourier Transform (STFT), Ridgelet Transform, Wavelet Transform (multi-level decomposition)

-Metrics: high-frequency energy ratio, gradient entropy, brightness/contrast stats

##### -Batch Processing

-Save all transformed images, wavelet coefficient maps, metrics (metrics.json), and summaries per image pair

-Auto-generate comparison panels for side-by-side visual inspection

-Optional PDF report compiling all panels

#### Workflow
-Download & Prepare Data

python fetch_and_prepare.py --num 80 --downscale 0.5 --blur 1.2 --L-scale 0.7
-Run Batch Analysis

python main_batch.py --pairs-csv data/pairs.csv --out-root results_batch
-Generate Comparison Panels

python make_panels.py --root results_batch


### 1. `channel_analysis_and_enhancement.py`

Covers basic color and geometric processing on the image `balloons.jpg`.

- RGB channel extraction and visualization
- YUV color space conversion
- Grayscale conversion with custom weighting
- Region flipping and rotation (manual)
- Color channel filtering (e.g., red emphasis)
- Image enhancement (e.g., HSV)

### 2. `histogram_and_contrast_enhancement.py`

Focuses on grayscale contrast enhancement for `q2.jpg`.

- Histogram computation and plotting
- Full-Scale Contrast Stretching (FSCS)
- Logarithmic contrast compression
- Gamma correction with multiple values
- Comparative analysis of contrast enhancement techniques

---

## 💻 Requirements

```bash
Python >= 3.7
numpy
matplotlib
opencv-python
