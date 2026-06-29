# TraceLens AI - Dataset Forensic Audit Report

Generated on: 2026-06-23 14:13:50
Total Images Audited: 9798
Audit Duration: 45.85 seconds

## 1. Key Statistics
- **Total REAL count**: 5398
- **Total FAKE count**: 4400
- **Average file size**: 18.23 KB
- **Average resolution**: 256x256
- **EXIF availability rate**: 51.63%

## 2. Resolution Distribution
- **Low Resolution (<256x256)**: 0
- **Medium Resolution (256x256 to 1024x1024)**: 9798
- **High Resolution (>1024x1024)**: 0
- **Extremely Small Images (<64x64)**: 0

## 3. Camera Make & Model Distribution
- **Total unique camera models**: 24
- **Top Camera Makes**:
  - Unknown: 4283
  - FUJIFILM: 321
  - Canon: 260
  - NIKON CORPORATION: 165
  - SONY: 30

- **Top Camera Models**:
  - Unknown: 4283
  - FinePix F100fd: 229
  - Canon PowerShot S3 IS: 115
  - NIKON D700: 64
  - FinePix F60fd: 61
  - NIKON D80: 50
  - Canon EOS 20D: 45
  - FinePix S3Pro: 31
  - Canon EOS 30D: 31
  - NIKON D200: 29

## 4. OOD & Ingestion Source Categorization
- **iPhone Photos**: 1049
- **Android Photos**: 1049
- **DSLR Photos**: 1100
- **Screenshots**: 1100
- **WhatsApp Compressed**: 1100
- **Instagram Compressed**: 0
- **Midjourney**: 1100
- **Flux**: 1100
- **ChatGPT**: 1100
- **Stable Diffusion (SDXL)**: 1100

## 5. Duplicate and Leakage Audit
- **Exact duplicate files**: 0 (0.00%)
- **Train/Test Leakage (SHA-256 overlap)**: 0 (0.00%)

## 6. Dataset Quality Gate (Phase 0.5 Status)
**Overall Status**: PASSED

- **gate_1_real_images_gt_3000**: PASS
- **gate_2_fake_images_gt_3000**: PASS
- **gate_3_smartphone_gt_1000**: PASS
- **gate_4_dslr_gt_500**: PASS
- **gate_5_screenshot_gt_500**: PASS
- **gate_6_leakage_eq_0**: PASS
- **gate_7_duplicate_lt_1pct**: PASS
- **gate_8_exif_report_generated**: PASS
- **gate_9_avg_res_gt_256**: PASS
- **gate_10_unique_camera_models_gt_5**: PASS
