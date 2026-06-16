# TraceLens Forensic Validation Report
### Benchmark Dataset: CASIA 2.0 (Au vs Tp)

> [!NOTE]
This validation report was automatically generated on the CASIA 2.0 forensic dataset. It evaluates the detection accuracy, threshold boundaries, and correlation coefficients of the TraceLens image integrity pipeline.

## 1. Executive Summary: Benchmark Metrics
The table below presents the validation metrics suitable for academic and internship presentations:

| Metric | Value |
| :--- | :--- |
| **Dataset Size** | 2000 images (Au/Clean: 1000, Tp/Tampered: 1000) |
| **True Positives (TP)** | 11 |
| **True Negatives (TN)** | 1000 |
| **False Positives (FP)** | 0 |
| **False Negatives (FN)** | 989 |
| **Accuracy** | 50.55% |
| **Precision** | 100.00% |
| **Recall (Sensitivity)** | 1.10% |
| **F1 Score** | 2.18% |
| **ROC-AUC Score** | 0.2831 |

## 2. Current Classification Method
This benchmark evaluates the **existing rule-based cumulative heuristic engine** of TraceLens and **not a machine-learning classifier**. Predictions are derived via the following explicit heuristics:

### Manipulation Risk Score Heuristic
The manipulation risk score starts at `0` (clean base) and cumulatively increments when specific forensic indicators are flagged:
* **Crop Detected**: `+15` points
* **Resize Detected**: `+10` points
* **Watermark Detected**: `+20` points
* **Recompression/Quantization Detected**: `+25` points
* **Screenshot Properties Detected**: `+15` points
* **Metadata Stripped Detected**: `+10` points

The accumulated score is capped within bounds of `0` to `95` points:
$$\text{Risk Score} = \max\left(0, \min\left(95, \sum \text{Weights of Active Triggers}\right)\right)$$

### Decision Boundary
An image is classified as **Manipulated (Label 1)** if its risk score is greater than **35** (which triggers a verdict of `Manipulated` or `Highly Suspicious` in the Media Profile):
$$\text{Prediction} = \begin{cases} 1 & \text{if } \text{manipulation\_risk} > 35 \\ 0 & \text{otherwise} \end{cases}$$

## 3. Forensic Indicator Contributions (Feature Importance)
To determine which forensic indicators contribute most strongly to the classification of manipulated assets, we compute the Pearson correlation coefficient ($r$) between each extracted indicator and the binary ground truth label. A higher absolute correlation indicates a stronger contribution to positive manipulation detection:

| Rank | Forensic Indicator | Pearson Correlation ($r$) | Contribution Strength |
| :--- | :--- | :--- | :--- |
| 1 | `stego_suspicion` | -0.5018 | Weak Negative |
| 2 | `screenshot_probability` | -0.3491 | Weak Negative |
| 3 | `manipulation_risk` | -0.3112 | Weak Negative |
| 4 | `compression_status_val` | -0.0937 | Weak Negative |
| 5 | `metadata_trust_score` | -0.0668 | Weak Negative |
| 6 | `blockiness` | +0.0169 | Negligible |
| 7 | `ai_generation_probability` | +0.0100 | Negligible |
| 8 | `crop_detected` | +0.0000 | Negligible |
| 9 | `resize_detected` | +0.0000 | Negligible |
| 10 | `watermark_detected` | +0.0000 | Negligible |

## 4. Decision Boundary Threshold Scan
The table below scans the threshold values for the `manipulation_risk` score from 5 to 95 to locate the mathematically optimal decision boundary for CASIA 2.0:

| Threshold | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 **(Optimum)** | 858 | 93 | 907 | 142 | 47.5% | 48.6% | 85.8% | 62.1% |
| 10 | 622 | 94 | 906 | 378 | 35.8% | 40.7% | 62.2% | 49.2% |
| 15 | 214 | 144 | 856 | 786 | 17.9% | 20.0% | 21.4% | 20.7% |
| 20 | 214 | 144 | 856 | 786 | 17.9% | 20.0% | 21.4% | 20.7% |
| 25 | 195 | 1000 | 0 | 805 | 59.8% | 100.0% | 19.5% | 32.6% |
| 30 | 195 | 1000 | 0 | 805 | 59.8% | 100.0% | 19.5% | 32.6% |
| 35 | 11 | 1000 | 0 | 989 | 50.5% | 100.0% | 1.1% | 2.2% |
| 40 | 10 | 1000 | 0 | 990 | 50.5% | 100.0% | 1.0% | 2.0% |
| 45 | 10 | 1000 | 0 | 990 | 50.5% | 100.0% | 1.0% | 2.0% |
| 50 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 55 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 60 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 65 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 70 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 75 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 80 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 85 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 90 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |
| 95 | 0 | 1000 | 0 | 1000 | 50.0% | 0.0% | 0.0% | 0.0% |

The mathematically optimal threshold is **5**, which achieves an F1 score of **62.06%** (compared to the current hardcoded threshold of 35 which yields an F1 score of **2.18%**).

## 5. ROC Curve Points
The Receiver Operating Characteristic (ROC) curve evaluates TPR (Sensitivity) against FPR (1 - Specificity) across decision thresholds:

| Threshold Scan | False Positive Rate (FPR) | True Positive Rate (TPR) |
| :--- | :--- | :--- |
| -1.0 | 1.0000 | 1.0000 |
| 0.0 | 0.9070 | 0.8580 |
| 10.0 | 0.9060 | 0.6220 |
| 15.0 | 0.8560 | 0.2140 |
| 25.0 | 0.0000 | 0.1950 |
| 35.0 | 0.0000 | 0.0110 |
| 40.0 | 0.0000 | 0.0100 |
| 50.0 | 0.0000 | 0.0000 |
| 51.0 | 0.0000 | 0.0000 |

**Area Under the ROC Curve (ROC-AUC)**: `0.2831`

## 6. False Positive Analysis
A **False Positive (FP)** occurs when an authentic (unmodified) image from the `Au` folder is classified as manipulated because its risk score exceeded 35.

No False Positives were detected in this evaluation run.

## 7. False Negative Analysis
A **False Negative (FN)** occurs when a tampered (manipulated) image from the `Tp` folder is classified as authentic because its risk score was 35 or lower.

Total False Negatives: 989 images.
### Key Root Causes Identified:
1. **Intact Provenance Signatures**: 0/989 False Negatives retained valid EXIF camera/capture signatures from their source files, meaning no metadata stripped penalty was triggered.
2. **Undetected Edge Blur / Resampling**: 857/989 False Negatives had a blockiness index under `1.2` (average JPEG), which failed to trigger compression/quantization penalties.
3. **Missing Copy-Move Ground Truth**: The rule-based engine evaluates local compression blockiness inconsistencies. If a tampered crop was perfectly resampled and recompressed, visual blockiness metrics remain uniform, leaving crop/splice artifacts undetected without semantic reference comparison.

#### Sample False Negative Files:
| Filename | Risk Score | Metadata Trust | Blockiness | Stego Suspicion |
| :--- | :---: | :---: | :---: | :---: |
| `Tp_D_CND_S_N_txt00028_txt00006_10848.jpg` | 15 | 10 | 1.04 | 10 |
| `Tp_D_CNN_M_B_nat00056_nat00099_11105.jpg` | 15 | 10 | 1.18 | 10 |
| `Tp_D_CND_M_N_art00076_art00077_10289.tif` | 35 | 15 | 1.12 | 0 |
| `Tp_D_CND_M_N_art00077_art00076_10290.tif` | 35 | 15 | 1.05 | 0 |
| `Tp_D_CND_S_N_ani00073_ani00068_00193.tif` | 35 | 15 | 1.11 | 0 |
| `Tp_D_CND_M_N_ani00018_sec00096_00138.tif` | 35 | 15 | 1.02 | 0 |
| `Tp_D_CND_S_N_ind00078_ind00077_00476.tif` | 10 | 15 | 1.14 | 0 |
| `Tp_D_CNN_M_B_nat10139_nat00059_11949.jpg` | 15 | 10 | 1.16 | 10 |

