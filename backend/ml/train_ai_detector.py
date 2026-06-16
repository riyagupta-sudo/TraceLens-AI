#!/usr/bin/env python3
"""
TraceLens AI - Deep Learning Forensic Engine
Script: train_ai_detector.py
Description: Production-ready PyTorch binary classifier leveraging EfficientNet-B0 
             to detect AI-generated imagery versus real/authentic media.
"""

import os
import sys
import time
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

# ==========================================
# 1. SETUP LOGGING & CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

CONFIG = {
    "train_dir": "dataset/ai_detection/train",
    "test_dir": "dataset/ai_detection/test",
    "model_save_path": "backend/models/ai_detector.pth",
    "curves_save_path": "backend/models/metrics_curves.png",
    "image_size": 64,
    "batch_size": 32,
    "epochs": 1,
    "lr": 1e-4,
    "patience": 5, # Early stopping patience
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Ensure destination directories exist
os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)


# ==========================================
# 2. DATA AUGMENTATION & TRANSFORMATIONS
# ==========================================
# ImageNet normalization coefficients utilized by timm models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]),
    "test": transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
}


# ==========================================
# 3. EARLY STOPPING CLASS
# ==========================================
class EarlyStopping:
    """Stops training early if validation loss doesn't improve after a specified patience."""
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            logging.info(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ==========================================
# 4. TRAINING & VALIDATION FUNCTION
# ==========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, config):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    early_stopping = EarlyStopping(patience=config["patience"])

    for epoch in range(1, config["epochs"] + 1):
        logging.info(f"Epoch {epoch}/{config['epochs']}")
        logging.info("-" * 25)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Batch"):
                inputs = inputs.to(config["device"])
                # Convert labels to float for BCEWithLogitsLoss and fix dimensions
                labels = labels.to(config["device"]).float().unsqueeze(1)

                optimizer.zero_grad()

                # Forward pass tracking history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Convert logits to binary predictions (threshold = 0.0 for logits)
                    preds = (outputs >= 0.0).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            # Deep copy the model weights if validation loss improves
            if phase == 'test':
                early_stopping(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, config["model_save_path"])
                    logging.info(f"--> Saved improved checkpoint to {config['model_save_path']}")

        print()
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Terminating training loop.")
            break

    time_elapsed = time.time() - since
    logging.info(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best Validation Loss: {best_loss:.4f}")

    # Load best model weights before final evaluation returning
    model.load_state_dict(best_model_wts)
    return model, history


# ==========================================
# 5. FINAL SCIENTIFIC EVALUATION METRICS
# ==========================================
def evaluate_final_performance(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Final Inference Engine"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Extract probabilities using standard sigmoid conversion
            probs = torch.sigmoid(outputs)
            preds = (outputs >= 0.0).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().squeeze(1).numpy())
            all_probs.extend(probs.cpu().squeeze(1).numpy())

    # Calculate requested performance parameters
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    # Output formatted report
    print("\n" + "="*45)
    print("   TRACELENS AI FORENSIC PERFORMANCE AUDIT   ")
    print("="*45)
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Precision:   {precision * 100:.2f}%")
    print(f"Recall:      {recall * 100:.2f}%")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print("-"*45)
    print("Confusion Matrix:")
    print(f"  True Negative (Real predicted as Real): {cm[0][0]}")
    print(f"  False Positive (Real predicted as Fake): {cm[0][1]}")
    print(f"  False Negative (Fake predicted as Real): {cm[1][0]}")
    print(f"  True Positive (Fake predicted as Fake): {cm[1][1]}")
    print("="*45 + "\n")


# ==========================================
# 6. PLOT LOSS CURVES
# ==========================================
def save_metrics_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    # Loss Curve subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Curve subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"Loss/Accuracy curves saved to {save_path}")


# ==========================================
# 7. EXECUTION PIPELINE ENTRYPOINT
# ==========================================
def main():
    logging.info(f"Execution initialized on target compute platform: {CONFIG['device']}")

    # Setup PyTorch Dataset ImageFolders
    try:
        image_datasets = {
            "train": datasets.ImageFolder(CONFIG["train_dir"], transform=data_transforms["train"]),
            "test": datasets.ImageFolder(CONFIG["test_dir"], transform=data_transforms["test"])
        }
    except Exception as e:
        logging.error(f"Failed to mount dataset partitions. Verify folder structural boundaries: {e}")
        return

    # Verify Class Assignments mapping (e.g., real=0, fake=1)
    class_mapping = image_datasets["train"].class_to_idx
    logging.info(f"Extracted Class Label Mappings: {class_mapping}")

    # Build Dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=CONFIG["batch_size"], shuffle=(x == 'train'), num_workers=0, pin_memory=False)
        for x in ['train', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Pull Pretrained EfficientNet-B0 backbone using timm
    logging.info("Initializing Pretrained EfficientNet-B0 backbone weights...")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
    model = model.to(CONFIG["device"])

    # Configure criteria components (BCEWithLogitsLoss handles numerical stability directly)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Execute training loop
    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        config=CONFIG
    )

    # Plot graphs and run thorough forensic reporting metrics
    save_metrics_curves(history, CONFIG["curves_save_path"])
    evaluate_final_performance(model, dataloaders["test"], CONFIG["device"])

if __name__ == '__main__':
    main()