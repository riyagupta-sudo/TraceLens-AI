# TraceLens AI – Known Limitations & Sensitivity Analysis
**AI Detector V1 Pipeline Optimization**

This document outlines the known limitations, out-of-distribution behaviors, and specific edge cases identified during optimization and benchmarking of the TraceLens AI Detector V1 pipeline.

---

## 1. Domain Sensitivity and Shift Factors

### A. Logit Shift on Camera Captures
* **Sensitivity**: The base EfficientNet-B0 model was trained predominantly on low-resolution benchmark images. When processing high-resolution camera captures (iPhone, Android, DSLR), the model frequently outputs negative logits (ranging from `-10.0` to `-40.0`).
* **Impact**: Under raw sigmoid scaling, negative logits translate directly to a very high probability of being FAKE. This creates a severe false-positive bias on authentic photography.
* **Mitigation**: Temperature scaling (`V1_TEMP = 20.0`) and Logistic Regression weight balancing adjust this bias, but raw logit distributions remain highly sensitive to sensor quality and lens type.

### B. Missing EXIF Metadata
* **Sensitivity**: Photos sent via messaging apps (WhatsApp, Telegram) or downloaded from email clients (Gmail) automatically have their metadata headers stripped.
* **Impact**: The metadata parser identifies the absence of camera make/model tags as an anomaly, triggering an EXIF warning penalty. On authentic photos that have simply been shared online, this penalty inflates the AI score and can result in false positives.
* **Mitigation**: The system incorporates `metadata_stripped_possible` logic in its consensus rules to lessen the penalty weight when platform redistribution is detected.

---

## 2. Signal Trigger Limitations

### A. Spurious FFT Peaks (Screenshots and Typography)
* **Sensitivity**: High-contrast, sharp transitions—such as UI text, geometric borders in screenshots, or digital logos—introduce artificial periodic patterns in the frequency domain.
* **Impact**: The 2D FFT periodic spike counter (triggering on counts $> 15$) flags these sharp text transitions as generative deconvolution grid artifacts, resulting in false-positive AI indicators.
* **Mitigation**: Standard screen aspect ratios and OCR word checks are cross-referenced to help classify the asset as a screenshot rather than AI-generated.

### B. Flat Region Texture Smoothing
* **Sensitivity**: Naturally flat surfaces—such as clear skies, dark scenes, studio backgrounds, or graphics—lack high-frequency grain.
* **Impact**: The Laplacian variance check flags variances $< 5.0$ as AI-inpainting or artificial smoothing, causing false positives.
* **Mitigation**: Flat-region suppression gates check for combined low entropy ($< 1.8$) and low noise variance to suppress warnings on naturally flat areas.

### C. CASIA / Splicing Retention (False Negatives)
* **Sensitivity**: Spliced images (where a real object is pasted onto a real background) preserve authentic camera visual patterns and sensor grain across the majority of the canvas.
* **Impact**: The global neural AI detector may output a positive logit (low AI score), missing the localized edit. Furthermore, spliced files often retain their original EXIF headers.
* **Mitigation**: The Two-Stage localized AI Editing Forensics Engine (`ai_editing_engine.py`) runs ELA and sliding-window entropy sweeps to catch these localized anomalies.

---

## 3. System and Dependency Warnings

### OCR winsdk Module Dependency
* **Issue**: On Windows environments missing the `winsdk` python package, the OCR engine logs an error: `[OCR ERROR] Failed to perform OCR: No module named 'winsdk'`.
* **Impact**: Heuristic OCR text detection is bypassed. The pipeline handles this exception gracefully and defaults to placeholder/regex filename checks to identify screenshots, ensuring execution is not interrupted.
