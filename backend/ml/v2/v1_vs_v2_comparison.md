# TraceLens AI - Detector V1 vs V2 Comparison Report

This comparison report profiles the performance of the legacy AI Detector V1 and the new production-grade AI Detector V2 (ConvNeXt-Tiny) on the isolated validation pack (375 images).

## 1. Executive Performance Summary

| Metric | Legacy V1 | Target Threshold | New V2 | Status |
| :--- | :---: | :---: | :---: | :---: |
| **iPhone False Positive Rate (FPR)** | 92.00% | < 10.0% | 20.00% | **FAILED** |
| **Android False Positive Rate (FPR)** | 92.00% | < 10.0% | 30.00% | **FAILED** |
| **DSLR False Positive Rate (FPR)** | 88.00% | < 10.0% | 28.00% | **FAILED** |
| **AI Recall (Sensitivity)** | 85.00% | > 85.0% | 65.00% | **FAILED** |
| **Overall F1-Score** | 64.27% | - | 70.27% | Improved |

---

## 2. Category Level Breakdown

Here are the detailed confusion matrices and performance metrics for each of the 9 categories in the validation manifest.

### REAL Categories (Clean Camera/Screenshot Files)

#### iPhone
- **Count**: 50
- **V1 (Legacy)**: FPR: 92.00%, Confusion Matrix: {'TN': 4, 'FP': 46, 'FN': 0, 'TP': 0}
- **V2 (New)**: FPR: 20.00%, Confusion Matrix: {'TN': 40, 'FP': 10, 'FN': 0, 'TP': 0}

#### Android
- **Count**: 50
- **V1 (Legacy)**: FPR: 92.00%, Confusion Matrix: {'TN': 4, 'FP': 46, 'FN': 0, 'TP': 0}
- **V2 (New)**: FPR: 30.00%, Confusion Matrix: {'TN': 35, 'FP': 15, 'FN': 0, 'TP': 0}

#### DSLR
- **Count**: 25
- **V1 (Legacy)**: FPR: 88.00%, Confusion Matrix: {'TN': 3, 'FP': 22, 'FN': 0, 'TP': 0}
- **V2 (New)**: FPR: 28.00%, Confusion Matrix: {'TN': 18, 'FP': 7, 'FN': 0, 'TP': 0}

#### Screenshots
- **Count**: 25
- **V1 (Legacy)**: FPR: 88.00%, Confusion Matrix: {'TN': 3, 'FP': 22, 'FN': 0, 'TP': 0}
- **V2 (New)**: FPR: 0.00%, Confusion Matrix: {'TN': 25, 'FP': 0, 'FN': 0, 'TP': 0}

#### WhatsApp
- **Count**: 25
- **V1 (Legacy)**: FPR: 92.00%, Confusion Matrix: {'TN': 2, 'FP': 23, 'FN': 0, 'TP': 0}
- **V2 (New)**: FPR: 32.00%, Confusion Matrix: {'TN': 17, 'FP': 8, 'FN': 0, 'TP': 0}

---

### FAKE Categories (AI Generated Files)

#### Midjourney
- **Count**: 50
- **V1 (Legacy)**: Precision: 100.00%, Recall: 88.00%, F1: 93.62%
- **V2 (New)**: Precision: 100.00%, Recall: 64.00%, F1: 78.05%

#### Flux
- **Count**: 50
- **V1 (Legacy)**: Precision: 100.00%, Recall: 76.00%, F1: 86.36%
- **V2 (New)**: Precision: 100.00%, Recall: 68.00%, F1: 80.95%

#### Stable Diffusion (SDXL)
- **Count**: 50
- **V1 (Legacy)**: Precision: 100.00%, Recall: 86.00%, F1: 92.47%
- **V2 (New)**: Precision: 100.00%, Recall: 60.00%, F1: 75.00%

#### ChatGPT
- **Count**: 50
- **V1 (Legacy)**: Precision: 100.00%, Recall: 90.00%, F1: 94.74%
- **V2 (New)**: Precision: 100.00%, Recall: 68.00%, F1: 80.95%
