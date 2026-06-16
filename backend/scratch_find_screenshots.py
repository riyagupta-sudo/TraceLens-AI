import os
import glob

# Search dataset and desktop
patterns = [
    "c:/Users/riya2/OneDrive/Desktop/TraceLens AI/dataset/**/*screenshot*",
    "c:/Users/riya2/OneDrive/Desktop/TraceLens AI/dataset/**/*.png",
    "c:/Users/riya2/OneDrive/Desktop/TraceLens AI/**/*Screenshot*",
    "c:/Users/riya2/OneDrive/Desktop/TraceLens AI/**/*.png"
]

for p in patterns:
    for filename in glob.glob(p, recursive=True):
        print(filename)
