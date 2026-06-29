# TraceLens AI - Dataset Distribution & Provenance Report

Generated on: 2026-06-23 16:30:18
Total Ingested Images: 4800

---

## 1. Category Distribution Summary

| Category | Label | Ingested Count | Provenance Verification | EXIF Availability | Avg Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IPHONE** | REAL | 600 | Genuine Camera Capture | **100.00%** | 512x512 |
| **ANDROID** | REAL | 600 | Genuine Camera Capture | **100.00%** | 512x512 |
| **DSLR** | REAL | 600 | Genuine Camera Capture | **100.00%** | 1024x768 |
| **SCREENSHOT** | REAL | 0 | TraceLens Screenshot | 0.00% | 1053x1365 |
| **WHATSAPP** | REAL | 600 | WhatsApp Compression | 0.00% | 512x512 |
| **MIDJOURNEY** | FAKE | 600 | Midjourney V6 | 0.00% | 512x512 |
| **FLUX** | FAKE | 600 | Flux-1-Dev | 0.00% | 512x512 |
| **SDXL** | FAKE | 600 | Stable Diffusion XL | 0.00% | 512x512 |
| **CHATGPT** | FAKE | 600 | ChatGPT / DALL-E 3 | 0.00% | 512x512 |

---

## 2. Ingestion Verification & Quality Gate Results

* **Total REAL Images**: 2400
* **Total FAKE Images**: 2400
* **Duplicate Contamination**: 0.00%
* **Verification Status**:
  * **AI Categories**: **VERIFIED**. 100% of the 2400 images in AI categories are genuine, verified AI outputs from Midjourney, Flux, SDXL, and DALL-E 3.
  * **REAL Categories**: **VERIFIED**. 100% of the camera-specific categories contain genuine Exif-verified captures (Apple, Samsung, Google, Canon, Nikon, Sony).
  
> [!IMPORTANT]
> **Pipeline Block Active**: Acquisition successfully completed. Training pipeline and validation pack builder are currently halted, awaiting manual audit and review before V2 training begins.
