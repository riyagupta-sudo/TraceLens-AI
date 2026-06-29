# TraceLens AI - Validation Pack Provenance Audit Report

Generated Audit of the validation pack datasets at `backend/ml/v2/validation_pack/`.

## 1. Summary of Provenance & Mislabeled Samples

* **Total REAL Images Copied**: 175
* **Total FAKE Images Copied**: 200
* **Number of Files in Manifest**: 375
* **Number of True Generative AI Images**: 0 (0.00%)
* **Number of Classical Manipulated Images (Splicing/Copy-Move)**: 200 (100.00% of FAKE class)
* **Number of Unknown Images**: 0

---

## 2. Category Assignment Confidence Breakdown

| Assigned Category | Verification Method & Source | Confidence | Sourced Dataset |
| :--- | :--- | :---: | :--- |
| **SCREENSHOT** | Sourced from `Screenshot/screenshot` folder | **HIGH** | TraceLens Screenshot Dataset |
| **WHATSAPP** | Sourced from `Screenshot/pictures` (WhatsApp compression) | **HIGH / LOW** | TraceLens Screenshot Dataset |
| **IPHONE** | EXIF check for Apple / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **ANDROID** | EXIF check for Android brands / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **DSLR** | EXIF check for Nikon, Canon, Sony, etc. / MD5 hash fallback | **HIGH / LOW** | CASIA v2 / TraceLens Dataset |
| **MIDJOURNEY** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **FLUX** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **SDXL** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |
| **CHATGPT** | MD5 hash modulo 4 fallback on CASIA tampered files | **LOW** | CASIA v2 Splicing Dataset |

---

## 3. Sample-Level Provenance Audit Records (First 50 Files)

Here are the details of the first 50 files from the validation pack:

| Filename | Label | Assigned Category | EXIF Make/Model | Verification Type | Original Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Au_ani_30324.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30324.jpg` |
| `Au_sec_30679.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_30679.jpg` |
| `Au_cha_30470.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_cha_30470.jpg` |
| `Au_nat_30450.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_nat_30450.jpg` |
| `Au_ani_30017.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30017.jpg` |
| `Au_ani_00003.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_00003.jpg` |
| `Au_sec_30456.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_30456.jpg` |
| `Au_ani_30305.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30305.jpg` |
| `Au_nat_20017.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_nat_20017.jpg` |
| `Au_arc_30610.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30610.jpg` |
| `Au_sec_30527.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_30527.jpg` |
| `Au_sec_20066.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_20066.jpg` |
| `Au_ani_30406.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30406.jpg` |
| `Au_sec_30284.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_30284.jpg` |
| `human_004.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals\human_004.jpg` |
| `Au_cha_30249.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_cha_30249.jpg` |
| `Au_arc_30683.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30683.jpg` |
| `Au_arc_30644.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30644.jpg` |
| `Au_arc_20028.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_20028.jpg` |
| `Au_ani_30217.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30217.jpg` |
| `Au_arc_30741.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30741.jpg` |
| `Au_arc_30716.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_arc_30716.jpg` |
| `Au_arc_30674.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30674.jpg` |
| `Au_arc_30046.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30046.jpg` |
| `Au_arc_30288.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30288.jpg` |
| `Au_arc_30098.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30098.jpg` |
| `Au_ani_30490.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30490.jpg` |
| `vehicle_003.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\originals\vehicle_003.jpg` |
| `Au_arc_30273.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30273.jpg` |
| `Au_ani_30312.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30312.jpg` |
| `Au_arc_30747.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30747.jpg` |
| `Au_arc_30417.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30417.jpg` |
| `Au_nat_30248.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_nat_30248.jpg` |
| `Au_arc_30665.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30665.jpg` |
| `Au_arc_20040.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_20040.jpg` |
| `Au_arc_30569.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30569.jpg` |
| `Au_arc_30470.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30470.jpg` |
| `Au_cha_30592.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_cha_30592.jpg` |
| `Au_arc_30289.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30289.jpg` |
| `Au_ani_30622.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30622.jpg` |
| `Au_ani_30516.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30516.jpg` |
| `Au_arc_30227.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30227.jpg` |
| `Au_cha_30669.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_cha_30669.jpg` |
| `Au_ani_30392.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30392.jpg` |
| `Au_sec_30661.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_sec_30661.jpg` |
| `Au_ani_30215.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_30215.jpg` |
| `Au_arc_30717.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30717.jpg` |
| `Au_ani_00017.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_ani_00017.jpg` |
| `Au_pla_30365.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary\authentic\Au_pla_30365.jpg` |
| `Au_arc_30284.jpg` | REAL | IPHONE | None | LOW (Hash fallback) | `C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\Screenshot\pictures\Au_arc_30284.jpg` |
