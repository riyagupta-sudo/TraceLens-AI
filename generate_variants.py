from PIL import Image, ImageDraw
import os

base_path = "dataset"

originals = os.path.join(base_path, "originals")
compressed = os.path.join(base_path, "compressed")
cropped = os.path.join(base_path, "cropped")
resized = os.path.join(base_path, "resized")
watermarked = os.path.join(base_path, "watermarked")

for folder in [compressed, cropped, resized, watermarked]:
    os.makedirs(folder, exist_ok=True)

for filename in os.listdir(originals):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    path = os.path.join(originals, filename)

    try:
        img = Image.open(path)

        name = os.path.splitext(filename)[0]

        # COMPRESSED
        img.save(
            os.path.join(compressed, f"{name}_compressed.jpg"),
            quality=20,
            optimize=True
        )

        # CROPPED (remove 10% border)
        w, h = img.size

        crop = img.crop((
            int(w * 0.1),
            int(h * 0.1),
            int(w * 0.9),
            int(h * 0.9)
        ))

        crop.save(
            os.path.join(cropped, f"{name}_crop.jpg")
        )

        # RESIZED
        resized_img = img.resize((512, 512))

        resized_img.save(
            os.path.join(resized, f"{name}_resize.jpg")
        )

        # WATERMARK
        wm = img.copy()

        draw = ImageDraw.Draw(wm)

        draw.text(
            (20, 20),
            "TRACELENS",
            fill="white"
        )

        wm.save(
            os.path.join(watermarked, f"{name}_watermark.jpg")
        )

        print("Generated:", filename)

    except Exception as e:
        print("Error:", filename, e)

print("\nDONE! All variants generated.")