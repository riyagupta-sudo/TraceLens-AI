import json
import os

project_root = r"c:\Users\riya2\OneDrive\Desktop\TraceLens AI"
v2_dir = os.path.join(project_root, "backend", "ml", "v2")
artifact_dir = r"C:\Users\riya2\.gemini\antigravity-ide\brain\3441a49b-e49e-4155-bb94-0c9acc60bea3"

def main():
    # JSON Data structure
    inventory_data = {
        "status": "Workspace contains no verified generative AI image datasets.",
        "search_scope": "Entire TraceLens AI workspace, including all directories and raw dataset directories.",
        "findings": {
            "total_genuine_ai_images_found": 0,
            "total_ai_metadata_sidecars_found": 0,
            "total_prompt_files_found": 0
        },
        "audited_categories": {
            "midjourney": {
                "folder_path": "N/A",
                "image_count": 0,
                "avg_resolution": "N/A",
                "file_formats": [],
                "metadata_availability": "N/A",
                "provenance_confidence": "LOW"
            },
            "flux": {
                "folder_path": "N/A",
                "image_count": 0,
                "avg_resolution": "N/A",
                "file_formats": [],
                "metadata_availability": "N/A",
                "provenance_confidence": "LOW"
            },
            "sdxl": {
                "folder_path": "N/A",
                "image_count": 0,
                "avg_resolution": "N/A",
                "file_formats": [],
                "metadata_availability": "N/A",
                "provenance_confidence": "LOW"
            },
            "chatgpt_dalle": {
                "folder_path": "N/A",
                "image_count": 0,
                "avg_resolution": "N/A",
                "file_formats": [],
                "metadata_availability": "N/A",
                "provenance_confidence": "LOW"
            }
        },
        "audited_workspace_folders": [
            {
                "directory_path": "dataset/ai_detection/train/fake",
                "image_count": 10000,
                "average_resolution": "32x32",
                "file_formats": ["JPEG"],
                "metadata_availability": "None (0.00% EXIF)",
                "provenance_confidence": "LOW",
                "suitability_for_ai_detection": "UNSUITABLE. Extremely low resolution (CIFAR-10 scale), out of distribution for modern images, lacks any generative AI artifacts.",
                "verified_image_type": "Classical low-resolution benchmark / CIFAR-10"
            },
            {
                "directory_path": "dataset/ai_detection_v2",
                "image_count": 4400,
                "average_resolution": "256x256",
                "file_formats": ["JPEG"],
                "metadata_availability": "None (0.00% EXIF)",
                "provenance_confidence": "LOW",
                "suitability_for_ai_detection": "UNSUITABLE. Contains crops of old CASIA v2 spliced/copy-moved photos, synthetically mislabeled with generator suffixes via filename hash modulo.",
                "verified_image_type": "Classical digital image manipulation / splicing"
            },
            {
                "directory_path": "dataset/casia_binary/tampered",
                "image_count": 399,
                "average_resolution": "451x337",
                "file_formats": ["JPEG"],
                "metadata_availability": "Partial (17.04% EXIF)",
                "provenance_confidence": "LOW",
                "suitability_for_ai_detection": "UNSUITABLE. Classical image splicing/copy-move digital manipulation database from 2007-2013.",
                "verified_image_type": "Classical digital image manipulation / splicing"
            }
        ],
        "conclusion": "TraceLens currently possesses ZERO verified AI-generated images. It is impossible to train a production-grade AI Detector V2 without first ingesting genuine generative AI image datasets."
    }

    # MD Data structure
    md_content = """# TraceLens AI - Dataset Source Discovery Audit Report

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
"""

    # Write to v2 directory
    with open(os.path.join(v2_dir, "ai_dataset_inventory.json"), "w") as f:
        json.dump(inventory_data, f, indent=4)
    with open(os.path.join(v2_dir, "ai_dataset_inventory.md"), "w") as f:
        f.write(md_content)

    # Write to artifacts directory
    with open(os.path.join(artifact_dir, "ai_dataset_inventory.json"), "w") as f:
        json.dump(inventory_data, f, indent=4)
    with open(os.path.join(artifact_dir, "ai_dataset_inventory.md"), "w") as f:
        f.write(md_content)

    print("Inventory files written successfully to both V2 and Artifacts directories.")

if __name__ == "__main__":
    main()
