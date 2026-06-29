# TraceLens AI – AI Detector V1 Pipeline Improvement Report

## 1. Validation Dataset Auto-Audit & Limitations
Prior to running calibration, an automatic audit of the validation set manifest `validation_manifest.json` was conducted.

### Key Finding:
- **Genuine AI-Generated Images**: **0 files** present in the `FAKE` partition.
- **Classical CASIA Spliced/Copy-Paste Images**: **200 files** (100.00% of the FAKE class).
- **Metadata Discrepancy**: Although the validation manifest labeled files under sources like `MIDJOURNEY` or `FLUX`, their file names (beginning with `Tp_`) and paths point exclusively to classical CASIA tampered images.
- **Adjustment**: Optimizations, ROC curves, and recall claims in this report are based honestly on **Classical CASIA Spliced/Tampered Detection** instead of modern generative AI recall.

---

## 2. Before vs. After Performance Comparison

| Metric | Before Improvement (V1 Base) | After Improvement (V1 Improved) |
| :--- | :---: | :---: |
| **Optimal Operating Threshold** | 0.5000 (Raw) | 0.4200 (Fused) |
| **Optimal Temperature T** | 8.0000 | 20.0000 |
| **Overall Accuracy** | 49.60% | 56.53% |
| **Precision** | 51.67% | 55.46% |
| **Recall (Tampered/CASIA)** | 85.00% | 94.00% |
| **F1-Score** | 0.6427 | 0.6976 |
| **ROC-AUC** | 0.4847 | 0.5936 |
| **Calibration Error (ECE)** | 0.3062 | 0.0086 |
| **Brier Score** | 0.3572 | 0.2336 |

---

## 3. Comparison of Fusion Strategies

1. **Strategy A (Temperature Scaling Only)**:
   - Sigmoid probabilities: prob = 1.0 - sigmoid(logit / 20.00)
   - Accuracy: 49.60%, ECE: 0.1699
2. **Strategy B (Weighted Linear Fusion)**:
   - Linear addition: p_fused = p_base + (-0.15) * I_exif + (0.0) * I_noise + (0.0) * I_fft
   - Accuracy: 53.07%, ECE: 0.0778
3. **Strategy C (Logistic Regression Fusion)**:
   - Logit formula: z = 0.0145 * (logit / 20.00) + -1.4883 * I_exif + 0.0000 * I_noise + -0.6455 * I_fft + -0.4134 * I_block + 1.5589
   - Accuracy: 56.53%, ECE: 0.0086

**Selected Strategy**: **Strategy C** is selected as it achieves the best balance of classification accuracy and calibration error.

---

## 4. Fitted Parameters for Production Integration

Save these parameters inside `dna_engine.py`:
- `V1_TEMP = 20.000000`
- `V1_THRESHOLD = 0.420000`
- `W_LOGIT = 0.014536`
- `W_EXIF = -1.488286`
- `W_NOISE = 0.000000`
- `W_FFT = -0.645450`
- `W_BLOCK = -0.413436`
- `LR_INTERCEPT = 1.558882`
