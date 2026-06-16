import torch
import torch.nn as nn
import timm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# ==========================
# CONFIG
# ==========================

DATA_DIR = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\dataset\casia_binary"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
IMAGE_SIZE = 128
# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

# ==========================
# TRANSFORMS
# ==========================

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================
# DATASET
# ==========================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

print("\nClasses:", dataset.classes)
print("Total Images:", len(dataset))

# ==========================
# TRAIN / TEST SPLIT
# ==========================

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

print(f"\nTrain Images: {len(train_dataset)}")
print(f"Test Images : {len(test_dataset)}")

# ==========================
# DATALOADERS
# ==========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# MODEL
# ==========================

print("\nLoading EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last feature block
for param in model.blocks[-1].parameters():
    param.requires_grad = True

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

print("\nTrainable Parameters:")

count = 0

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        count += param.numel()

print("\nTotal Trainable:", count)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# ==========================
# TRAINING
# ==========================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx}/{len(train_loader)}"
            )

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}%\n"
    )

# ==========================
# EVALUATION
# ==========================

print("\nEvaluating Model...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total

print(f"\nTest Accuracy: {test_acc:.2f}%")

# ==========================
# SAVE MODEL
# ==========================

SAVE_PATH = r"C:\Users\riya2\OneDrive\Desktop\TraceLens AI\backend\models\casia_detector.pth"

torch.save(
    model.state_dict(),
    SAVE_PATH
)

print(f"\nModel Saved:")
print(SAVE_PATH)

print("\nTraining Complete!")