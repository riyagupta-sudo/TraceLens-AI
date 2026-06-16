from PIL import Image, ImageDraw

# Create a mock screenshot of size 1920x1080
width = 1920
height = 1080
img = Image.new("RGB", (width, height), color=(18, 18, 24))

# Draw some mock text
draw = ImageDraw.Draw(img)
text = "TraceLens System Diagnostic Logs - Case ID 12804\nUser: investigator@tracelens.internal\nStatus: Scanning system memory...\nCompleted 98% of target acquisition."
draw.text((50, 50), text, fill=(0, 229, 255))

# Save image as PNG to the dataset folder
img.save("c:/Users/riya2/OneDrive/Desktop/TraceLens AI/dataset/Screenshot 2026-06-09.png", "PNG")
print("Screenshot generated at c:/Users/riya2/OneDrive/Desktop/TraceLens AI/dataset/Screenshot 2026-06-09.png")
