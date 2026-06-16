from PIL import Image
import imagehash
import csv
import os

base = "dataset"

originals = os.path.join(base, "originals")

variants = {
    "compressed": os.path.join(base, "compressed"),
    "cropped": os.path.join(base, "cropped"),
    "resized": os.path.join(base, "resized"),
    "watermarked": os.path.join(base, "watermarked")
}

with open("results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "original",
        "variant",
        "type",
        "phash_distance",
        "dhash_distance",
        "ahash_distance"
    ])

    for file in os.listdir(originals):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        original_path = os.path.join(originals, file)

        original_img = Image.open(original_path)

        p1 = imagehash.phash(original_img)
        d1 = imagehash.dhash(original_img)
        a1 = imagehash.average_hash(original_img)

        name = os.path.splitext(file)[0]

        for variant_type, folder in variants.items():

            if variant_type == "compressed":
                variant_name = f"{name}_compressed.jpg"

            elif variant_type == "cropped":
                variant_name = f"{name}_crop.jpg"

            elif variant_type == "resized":
                variant_name = f"{name}_resize.jpg"

            else:
                variant_name = f"{name}_watermark.jpg"

            variant_path = os.path.join(folder, variant_name)

            if not os.path.exists(variant_path):
                continue

            variant_img = Image.open(variant_path)

            p2 = imagehash.phash(variant_img)
            d2 = imagehash.dhash(variant_img)
            a2 = imagehash.average_hash(variant_img)

            writer.writerow([
                file,
                variant_name,
                variant_type,
                p1 - p2,
                d1 - d2,
                a1 - a2
            ])

print("CSV report generated successfully!")