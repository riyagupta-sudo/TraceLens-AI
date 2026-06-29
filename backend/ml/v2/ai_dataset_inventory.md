# TraceLens AI - Dataset Source Discovery Audit Report

## Executive Summary
> [!IMPORTANT]
> **Workspace contains no verified generative AI image datasets.**
> A complete workspace-wide audit was conducted. TraceLens currently possesses **zero** genuine generative AI images (Midjourney, Flux, SDXL, or ChatGPT/DALL-E) and **zero** corresponding prompt sidecars or manifests.

---

## 1. Audited Workspace Folders

| Directory Path | Image Count | Avg Resolution | File Formats | EXIF / Metadata | Provenance Confidence | Verified Image Type & Suitability |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`dataset/ai_detection/train/fake`** | 10,000 | 32x32 | JPEG | 0.00% | **LOW** | **CIFAR-10 / Low-res benchmark**. Completely unsuitable for modern AI detector training due to resolution scale. |
| **`dataset/ai_detection_v2`** | 4,400 | 256x256 | JPEG | 0.00% | **LOW** | **Classical Splicing Crops**. Cropped patches of 2010 CASIA splicing database, synthetically renamed with generator tags. |
| **`dataset/casia_binary/tampered`** | 399 | 451x337 | JPEG | 17.04% | **LOW** | **Classical Splicing Database**. Legacy Photoshop splicing edits from 2007-2013. |

---

## 2. Discovery Status by AI Generator Category

* **Midjourney**: **0 images** found in workspace.
* **Flux**: **0 images** found in workspace.
* **SDXL**: **0 images** found in workspace.
* **Stable Diffusion**: **0 images** found in workspace.
* **DALL-E / ChatGPT**: **0 images** found in workspace.

---

## 3. Conclusion & Recommendation

TraceLens currently **does not** possess sufficient verified AI-generated images to build a production AI Detector V2.
The legacy V1 model was trained on 32x32 pixel images (CIFAR-10 scale), which explain its high false-positive rates on modern high-resolution photographs.

**Remediation Recommendation**: Ingest genuine generative AI datasets (such as **GenImage** or **DiffusionDB**) containing verified high-resolution outputs from Midjourney, Flux, SDXL, and DALL-E 3 before attempting V2 training.
