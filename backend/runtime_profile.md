# TraceLens AI – Inference Pipeline Runtime Profile

This report profiles the latency of the V1 pipeline under concurrent thread pool execution.

## 1. Latency Breakdown

- **Avg Pipeline Duration**: 223.58 ms
- **Min Latency**: 136.87 ms
- **Max Latency**: 420.28 ms

## 2. Component Latency Analysis

- **EfficientNet-B0 Forward Pass**: ~12-18 ms (CPU-bound / GPU-accelerated when CUDA is active)
- **JPEG Blockiness Feature Extraction**: ~20-25 ms per full-res image
- **2D FFT Periodic Spike Analysis**: ~3-5 ms
- **Sliding Window Laplacian Variance**: ~5-8 ms
