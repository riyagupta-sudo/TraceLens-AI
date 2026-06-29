# TraceLens AI Detector V1 – Baseline Benchmarking Report

This report presents the performance of the **Original V1 (Legacy)** and **Improved V1 (Logistic Regression)** pipelines on multiple generalization datasets.

## Dataset: Validation Pack

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 50.13% | 51.91% | 88.50% | 0.6543 | 93.71% | 0.3658 | 0.3995 |
| Improved V1 (LogReg) | 56.53% | 55.46% | 94.00% | 0.6976 | 86.29% | 0.0086 | 0.2336 |

## Dataset: Smartphone Photos

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 4.00% | 0.00% | 0.00% | 0.0000 | 96.00% | 0.7162 | 0.5391 |
| Improved V1 (LogReg) | 98.00% | 0.00% | 0.00% | 0.0000 | 2.00% | 0.4185 | 0.1766 |

## Dataset: DSLR Photos

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 4.00% | 0.00% | 0.00% | 0.0000 | 96.00% | 0.7521 | 0.5908 |
| Improved V1 (LogReg) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.4143 | 0.1717 |

## Dataset: WhatsApp Images

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 4.00% | 0.00% | 0.00% | 0.0000 | 96.00% | 0.7202 | 0.5447 |
| Improved V1 (LogReg) | 96.00% | 0.00% | 0.00% | 0.0000 | 4.00% | 0.4156 | 0.1727 |

## Dataset: Gmail Images

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 0.00% | 0.00% | 0.00% | 0.0000 | 100.00% | 0.7514 | 0.5839 |
| Improved V1 (LogReg) | 100.00% | 0.00% | 0.00% | 0.0000 | 0.00% | 0.4146 | 0.1719 |

## Dataset: Screenshot Images

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 8.00% | 0.00% | 0.00% | 0.0000 | 92.00% | 0.8019 | 0.6837 |
| Improved V1 (LogReg) | 2.00% | 0.00% | 0.00% | 0.0000 | 98.00% | 0.5186 | 0.2714 |

## Dataset: AI-Edited Images

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 92.00% | 100.00% | 92.00% | 0.9583 | 0.00% | 0.2190 | 0.0969 |
| Improved V1 (LogReg) | 94.00% | 100.00% | 94.00% | 0.9691 | 0.00% | 0.4367 | 0.2074 |

## Dataset: Variants (Crop/Resize/Compress)

| Pipeline | Accuracy | Precision | Recall | F1 Score | False Positive Rate | ECE | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Original V1 (Legacy) | 16.67% | 0.00% | 0.00% | 0.0000 | 83.33% | 0.7815 | 0.6639 |
| Improved V1 (LogReg) | 23.33% | 0.00% | 0.00% | 0.0000 | 76.67% | 0.4792 | 0.2340 |

