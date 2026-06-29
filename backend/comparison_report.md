# TraceLens AI Detector V1 – Pipeline Comparison Report

This report presents the comparative metrics of the three candidate fusion strategies across multiple generalization datasets.

## Dataset: Validation Pack

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 56.53% | 55.46% | 94.00% | 0.6976 | 86.29% | 0.0086 |
| Decision Tree (Fitted) | 76.53% | 73.53% | 87.50% | 0.7991 | 36.00% | 0.0000 |
| Weighted Linear (Fitted LR) | 68.27% | 73.68% | 63.00% | 0.6792 | 25.71% | 0.0785 |

## Dataset: Smartphone

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 98.00% | 0.00% | 0.00% | 0.0000 | 2.00% | 0.4185 |
| Decision Tree (Fitted) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0040 |
| Weighted Linear (Fitted LR) | 99.00% | 0.00% | 0.00% | 0.0000 | 1.00% | 0.0465 |

## Dataset: DSLR

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.4143 |
| Decision Tree (Fitted) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0040 |
| Weighted Linear (Fitted LR) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0353 |

## Dataset: WhatsApp

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 96.00% | 0.00% | 0.00% | 0.0000 | 4.00% | 0.4156 |
| Decision Tree (Fitted) | 98.00% | 0.00% | 0.00% | 0.0000 | 2.00% | 0.0173 |
| Weighted Linear (Fitted LR) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0374 |

## Dataset: Gmail

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.4146 |
| Decision Tree (Fitted) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0040 |
| Weighted Linear (Fitted LR) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.0355 |

## Dataset: Screenshots

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 2.00% | 0.00% | 0.00% | 0.0000 | 98.00% | 0.5186 |
| Decision Tree (Fitted) | 40.00% | 0.00% | 0.00% | 0.0000 | 60.00% | 0.4061 |
| Weighted Linear (Fitted LR) | 58.00% | 0.00% | 0.00% | 0.0000 | 42.00% | 0.5185 |

## Dataset: AI-Edited

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 94.00% | 100.00% | 94.00% | 0.9691 | 0.00% | 0.4367 |
| Decision Tree (Fitted) | 70.00% | 100.00% | 70.00% | 0.8235 | 0.00% | 0.4182 |
| Weighted Linear (Fitted LR) | 58.00% | 100.00% | 58.00% | 0.7342 | 0.00% | 0.4309 |

## Dataset: Variants

| Model | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 23.33% | 0.00% | 0.00% | 0.0000 | 76.67% | 0.4792 |
| Decision Tree (Fitted) | 8.33% | 0.00% | 0.00% | 0.0000 | 91.67% | 0.2519 |
| Weighted Linear (Fitted LR) | 20.00% | 0.00% | 0.00% | 0.0000 | 80.00% | 0.6701 |

