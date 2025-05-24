---

# 🗜️ Reduced Bit Median Quantization (RBMQ) for Efficient Image Compression

## 📌 Overview

Reduced Bit Median Quantization (RBMQ) is an innovative image preprocessing and standalone compression method that focuses on **bit reduction**, **value clustering**, and **redundancy enhancement** to facilitate **efficient image compression**. It is designed to work independently or as a preprocessing step to **enhance the performance of standard compressors** like **JPEG**, **PNG**, and **JPEG2000**.

RBMQ operates by segmenting the intensity range of images into **fixed-width bins** and replacing each pixel with a **median-based representative value**. This reduction in bit variability improves **storage efficiency**, **reduces entropy**, and **enhances compressibility** through downstream encoding techniques such as **Run-Length Encoding (RLE)** and **Huffman coding**.

---

## 🚀 Features

- 🔧 **Standalone Compression Method** — Reduce file size without relying on traditional compression algorithms.
- 📉 **Preprocessing for Standard Formats** — Improves the compression ratio of JPEG, PNG, and JPEG2000 formats.
- 🎯 **Bit Reduction** — Decreases bit-depth by replacing pixel values with representative medians.
- 🔁 **Redundancy Introduction** — Facilitates better compression with Huffman and RLE.
- 🖼️ **Supports Grayscale and RGB Images** — Handles both single- and multi-channel data.
- 📊 **Benchmarking Framework** — Compare with Uniform Quantization, JPEG, JPEG2000, and PNG under realistic conditions.

---

## 🧠 Theoretical Background

In traditional image compression, techniques like DCT or wavelet transforms are applied to reduce spatial redundancy. RBMQ takes a different approach:

- **Quantization Phase**: Intensity range [0–255] is divided into 32 segments (bins), each with 8 values.
- **Median Selection**: The **5th value** of each bin is selected as the representative median.
- **Pixel Replacement**: Each pixel is quantized to its bin's median value.

### 🎯 Key Concept

> RBMQ introduces **value clustering** without sophisticated transforms, yielding fewer unique values and lower entropy — critical for downstream encoders.

---

## 🗂️ Project Structure

```
rbmq-image-compression/
│
├── images/                # Raw input images for testing (grayscale/RGB)
├── quantized/             # RBMQ-processed outputs
├── results/               # Benchmark results and comparison tables
├── rbmq/                  # Core algorithm and helpers
│   ├── quantizer.py       # RBMQ logic for grayscale and RGB
│   ├── utils.py           # Helper functions for image I/O
│   └── metrics.py         # PSNR, SSIM, entropy, etc.
├── benchmarks/            # Scripts for JPEG, PNG, JPEG2000 compression
├── analysis/              # Scripts for CR, PSNR, SSIM evaluation
├── README.md              # Documentation
└── requirements.txt       # Dependencies
```

---

## 🖥️ Installation

```bash
git clone https://github.com/yourusername/rbmq-image-compression.git  
cd rbmq-image-compression
pip install -r requirements.txt
```

### 📦 Requirements

- `numpy`
- `opencv-python`
- `scikit-image`
- `imageio`
- `Pillow`
- `matplotlib`
- `pywavelets` (for JPEG2000 if needed)
- `imagecodecs` (alternative for JPEG2000)

---

## 📸 Sample Usage

### Apply RBMQ to a Grayscale Image:

```python
from rbmq.quantizer import apply_rbmq_grayscale
apply_rbmq_grayscale("images/lena_gray.bmp", "quantized/lena_rbmq.png")
```

### Apply RBMQ to an RGB Image:

```python
from rbmq.quantizer import apply_rbmq_rgb
apply_rbmq_rgb("images/lena_color.tiff", "quantized/lena_rbmq_color.png")
```

---

## 📈 Benchmarking

### Compared Methods:

- Uniform Quantization
- JPEG (DCT + Quantization + Huffman)
- JPEG2000 (Wavelet)
- PNG (Lossless Huffman + LZ77)
- RBMQ + PNG/JPEG

### Evaluation Metrics:

| Metric        | Description                              |
|---------------|------------------------------------------|
| Compression Ratio (CR) | File size reduction effectiveness |
| PSNR          | Peak Signal-to-Noise Ratio (measures distortion) |
| SSIM          | Structural Similarity Index (perceptual similarity) |
| Entropy       | Redundancy/variability of pixel data     |
| Execution Time | Speed of compression process            |

### Example Results Table:

| Image     | Method      | CR   | PSNR | SSIM | Entropy | Time (s) |
|-----------|-------------|------|------|------|---------|----------|
| lena.bmp  | JPEG        | 12.4 | 33.2 | 0.91 | 6.88    | 0.03     |
| lena.bmp  | PNG         | 8.7  | 42.5 | 0.99 | 7.15    | 0.05     |
| lena.bmp  | RBMQ + JPEG | 15.2 | 31.4 | 0.88 | 5.13    | 0.06     |
| lena.bmp  | RBMQ + PNG  | 11.6 | 41.1 | 0.97 | 5.22    | 0.07     |

---

## 📬 Applications

- 📂 Archival of medical and satellite images with limited bandwidth  
- 📤 Preprocessing for cloud image uploads to save storage  
- 🧠 Academic research in low-bit image representations  
- 🧮 Visual entropy reduction for simplified image classification tasks  

---

## 📚 Research Context

This project is a component of a broader study titled:

> “Reduced Bit Median Quantization for Efficient Image Compression”

The research investigates how reducing intensity levels via median-based quantization introduces structured redundancy, aiding downstream compressors. RBMQ is shown to be particularly effective when used with lossless PNG or lossy JPEG.

---

## 🧪 Future Work

- 🧬 Adaptive bin sizing based on local image statistics  
- 🧠 Integration with deep learning image codecs  
- 📉 Progressive compression layers using RBMQ as the base  

---

## 🤝 Contributing

We welcome contributions from the open-source community! Feel free to:

- Submit pull requests  
- Report issues  
- Suggest new features or datasets  

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📧 Contact

- **Author**: [Your Name]  
- **Email**: youremail@example.com  
- **Affiliation**: [Your Institution]  
- **Research Advisor**: [Advisor Name]  

🌟 If you find this project helpful, please consider starring the repo!

--- 