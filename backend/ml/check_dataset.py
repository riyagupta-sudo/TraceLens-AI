from torchvision import datasets

dataset = datasets.ImageFolder(
    r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary"
)

print("Classes:", dataset.classes)
print("Class Mapping:", dataset.class_to_idx)

authentic = sum(1 for _, y in dataset.samples if y == 0)
tampered = sum(1 for _, y in dataset.samples if y == 1)

print("Authentic:", authentic)
print("Tampered :", tampered)
print("Total    :", len(dataset))