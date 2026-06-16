# TraceLens Forensic Feature Audit Report
### Dataset: CASIA 2.0 Feature Vectors

## 1. Dataset Dimensions
* **Total Rows (Samples)**: `2000`
* **Total Columns (Features)**: `13`

## 2. Features Evaluated
The columns present in the audited feature vector dataset are:
* `filepath`
* `filename`
* `label`
* `manipulation_risk`
* `screenshot_probability`
* `metadata_trust_score`
* `blockiness`
* `crop_detected`
* `resize_detected`
* `watermark_detected`
* `compression_status`
* `ai_generation_probability`
* `stego_suspicion`

## 3. Data Quality & Missingness Analysis
Count of missing (`NaN`) values for each field in the dataset:

| Column Name | Missing Count |
| :--- | :---: |
| `filepath` | 0 |
| `filename` | 0 |
| `label` | 0 |
| `manipulation_risk` | 0 |
| `screenshot_probability` | 0 |
| `metadata_trust_score` | 0 |
| `blockiness` | 0 |
| `crop_detected` | 0 |
| `resize_detected` | 0 |
| `watermark_detected` | 0 |
| `compression_status` | 0 |
| `ai_generation_probability` | 0 |
| `stego_suspicion` | 0 |

## 4. Cardinality Analysis (Unique Counts)
Number of unique values in every column:

| Column Name | Unique Value Count |
| :--- | :---: |
| `filepath` | 2000 |
| `filename` | 2000 |
| `label` | 2 |
| `manipulation_risk` | 7 |
| `screenshot_probability` | 8 |
| `metadata_trust_score` | 7 |
| `blockiness` | 1931 |
| `crop_detected` | 1 |
| `resize_detected` | 1 |
| `watermark_detected` | 1 |
| `compression_status` | 3 |
| `ai_generation_probability` | 4 |
| `stego_suspicion` | 3 |

## 5. Feature Values Grouped by Label (Class Means)
Comparison of mean values between Authentic images (`label = 0`) and Tampered images (`label = 1`):

| Forensic Metric | Mean (Label 0 - Clean) | Mean (Label 1 - Tampered) | Diff (Tp - Au) |
| :--- | :---: | :---: | :---: |
| `manipulation_risk` | 22.1600 | 15.9350 | -6.2250 |
| `screenshot_probability` | 29.4300 | 23.5950 | -5.8350 |
| `metadata_trust_score` | 20.5150 | 18.2850 | -2.2300 |
| `blockiness` | 1.1032 | 1.1070 | +0.0038 |
| `crop_detected` | 0.0000 | 0.0000 | +0.0000 |
| `resize_detected` | 0.0000 | 0.0000 | +0.0000 |
| `watermark_detected` | 0.0000 | 0.0000 | +0.0000 |
| `ai_generation_probability` | 16.3560 | 16.5600 | +0.2040 |
| `stego_suspicion` | 12.4600 | 5.2000 | -7.2600 |

## 6. Random Forest Classifier Feature Importance
A `RandomForestClassifier` was trained on all numeric and mapped features (excluding path identifiers) to determine non-linear feature importances in classifying manipulation:

| Rank | Mapped Feature | Gini Importance (Gini Decrease) |
| :---: | :--- | :---: |
| 1 | `manipulation_risk` | 0.2871 |
| 2 | `metadata_trust_score` | 0.2367 |
| 3 | `stego_suspicion` | 0.1858 |
| 4 | `screenshot_probability` | 0.1382 |
| 5 | `compression_status_val` | 0.0741 |
| 6 | `blockiness` | 0.0627 |
| 7 | `ai_generation_probability` | 0.0155 |
| 8 | `crop_detected` | 0.0000 |
| 9 | `resize_detected` | 0.0000 |
| 10 | `watermark_detected` | 0.0000 |

### Interpretations:
1. **Stego Suspicion & Screenshot Probability**: These indicators dominate the decision trees due to the systematic differences in noise entropy and structure in authentic CASIA images compared to edited variants.
2. **Zero-Importance Heuristics**: Parent-dependent flags (`crop_detected`, `resize_detected`, `watermark_detected`) have exactly `0.0000` importance since their values are identical (all `0`) for both categories in a blind verification context.
