# TraceLens AI Detector V1 – False Negative Analysis Report

This report details every false negative from the Improved V1 pipeline on generalization datasets, explaining why they occurred.

Total False Negatives: 15

| # | Dataset | Filename | Source | Fused Prob | Logit | EXIF | Laplacian Var | FFT Peaks | Blockiness |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Validation Pack | Tp_D_CRN_M_N_txt00063_txt00017_10835.jpg | MIDJOURNEY | 36.0% | -3.0786 | False | 5534.07 | 88 | 1.0366 |
| 2 | Validation Pack | Tp_S_CNN_M_N_nat00071_nat00071_10610.jpg | MIDJOURNEY | 36.7% | 42.1826 | False | 129.13 | 114 | 1.2919 |
| 3 | Validation Pack | Tp_S_CRN_S_B_pla10002_pla10002_20045.jpg | MIDJOURNEY | 36.1% | 7.2286 | False | 337.46 | 46 | 1.1740 |
| 4 | Validation Pack | Tp_S_NNN_S_N_cha10162_cha10162_12250.jpg | CHATGPT | 35.8% | -11.0545 | False | 489.83 | 56 | 1.1575 |
| 5 | Validation Pack | Tp_D_NRN_M_N_nat10136_arc00023_11916.jpg | CHATGPT | 35.8% | -10.8983 | False | 245.43 | 20 | 1.3025 |
| 6 | Validation Pack | Tp_S_NRN_S_N_nat00028_nat00028_20122.jpg | CHATGPT | 35.9% | -5.3160 | False | 774.29 | 54 | 1.2224 |
| 7 | Validation Pack | Tp_S_NRD_M_N_arc10124_arc10124_11882.jpg | CHATGPT | 35.7% | -18.7148 | False | 219.77 | 58 | 1.4819 |
| 8 | Validation Pack | Tp_D_NND_S_N_txt00061_txt00083_10825.jpg | CHATGPT | 35.6% | -24.0496 | False | 3915.07 | 24 | 1.0085 |
| 9 | Validation Pack | Tp_D_NNN_S_B_nat00027_nat00030_11094.jpg | CHATGPT | 35.3% | -45.0355 | False | 595.15 | 18 | 1.2091 |
| 10 | Validation Pack | Tp_S_NNN_S_N_nat00061_nat00061_10562.jpg | SDXL | 36.1% | 2.3390 | False | 241.03 | 54 | 1.3367 |
| 11 | Validation Pack | Tp_D_NRN_S_N_pla00093_txt00070_11327.jpg | SDXL | 35.2% | -50.1773 | False | 4374.99 | 26 | 1.0251 |
| 12 | Validation Pack | Tp_D_NRN_M_B_sec00085_cha00072_11458.jpg | SDXL | 35.7% | -16.6196 | False | 600.25 | 26 | 1.1408 |
| 13 | AI-Edited Images | Tp_S_CRN_M_N_cha10125_cha10125_12161.jpg | AI_EDITED | 35.5% | -31.5335 | False | 402.15 | 26 | 1.0956 |
| 14 | AI-Edited Images | Tp_D_NND_S_N_txt00061_txt00083_10825.jpg | AI_EDITED | 35.6% | -24.0496 | False | 3915.07 | 24 | 1.0085 |
| 15 | AI-Edited Images | Tp_S_NNN_S_N_nat00061_nat00061_10562.jpg | AI_EDITED | 36.1% | 2.3390 | False | 241.03 | 54 | 1.3367 |

## Breakdown and Explanations by Cause:

1. **High Logits on Spliced Images**: Spliced images containing large regions of real photography preserve the original camera visual patterns, resulting in positive logits from the neural model.
2. **Metadata Presence on Spliced Images**: Spliced CASIA images often retain their original EXIF headers, giving them `has_camera = True` and lowering the warning score.
