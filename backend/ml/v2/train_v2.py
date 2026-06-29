#!/usr/bin/env python3
"""
TraceLens AI - Phase 2 AI Detector V2 Training Pipeline
Script: train_v2.py
Description: Advanced PyTorch training pipeline using EfficientNet-B3 at 224x224
             resolution, supporting mixed precision, TensorBoard logging, early stopping,
             calibration metrics generation, and a quick dry-run mode.
"""

import os
import sys
import time
import json
import argparse
import copy
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass
from torchvision import datasets, transforms

import timm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    roc_curve
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args():
    parser = argparse.ArgumentParser(description="TraceLens AI V2 Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run a quick dry-run training with small subset of data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone weights and only train head")
    parser.add_argument("--no-freeze", action="store_false", dest="freeze_backbone", help="Disable backbone weight freezing")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Fraction of dataset to use for training")
    return parser.parse_args()

class EarlyStopping:
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

def calculate_ece_and_bins(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    histogram_stats = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_upper)
            
        count = int(np.sum(in_bin))
        histogram_stats.append({
            "bin": i,
            "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
            "count": count
        })
        
        if count > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += count * np.abs(bin_acc - bin_conf)
            
            bin_stats.append({
                "bin": i,
                "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                "confidence": float(round(bin_conf, 4)),
                "accuracy": float(round(bin_acc, 4)),
                "count": count
            })
        else:
            bin_stats.append({
                "bin": i,
                "range": f"[{bin_lower:.1f}, {bin_upper:.1f})",
                "confidence": 0.0,
                "accuracy": 0.0,
                "count": 0
            })
            
    ece = float(ece / len(probs)) if len(probs) > 0 else 0.0
    return ece, bin_stats, histogram_stats

def main():
    args = parse_args()
    
    # Path Configuration
    v2_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(v2_dir)))
    
    # Programmatic Quality Gate Check
    readiness_report_path = os.path.join(v2_dir, "dataset_readiness_report.json")
    if not os.path.exists(readiness_report_path):
        logging.error("Quality Gate readiness report not found! Aborting training.")
        sys.exit(1)
        
    try:
        with open(readiness_report_path, 'r') as r_f:
            readiness_data = json.load(r_f)
            status = readiness_data.get("readiness_status", "FAILED")
            if status != "PASSED":
                logging.error(f"Quality Gate failed (Status: {status})! Aborting training.")
                logging.error(f"Recommendations: {readiness_data.get('remediation_recommendations', [])}")
                sys.exit(1)
            logging.info("Quality Gate check passed successfully. Proceeding with training.")
    except Exception as e:
        logging.error(f"Failed to parse Quality Gate readiness report: {e}. Aborting training.")
        sys.exit(1)
        
    train_dir = os.path.join(project_root, "dataset", "ai_detection_v2", "train")
    test_dir = os.path.join(project_root, "dataset", "ai_detection_v2", "test")
    
    model_save_path = os.path.join(v2_dir, "ai_detector_v2.pth")
    checkpoint_save_path = os.path.join(v2_dir, "checkpoint.pth")
    eval_json_path = os.path.join(v2_dir, "evaluation.json")
    calib_json_path = os.path.join(v2_dir, "model_calibration.json")
    tensorboard_dir = os.path.join(v2_dir, "runs")
    
    os.makedirs(v2_dir, exist_ok=True)
    
    # Image properties
    image_size = 224
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    
    if args.quick:
        logging.info("[QUICK RUN] Activating fast debugging mode...")
        epochs = 1
        batch_size = 4
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Target compute platform: {device}")
    
    # Define transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]),
        "test": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    }
    
    # Load Datasets
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
        test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    except Exception as e:
        logging.error(f"Failed to mount datasets. Using dummy datasets for quick dry run if enabled. Error: {e}")
        if args.quick:
            # Create a dummy folder structure for dry run testing
            os.makedirs(os.path.join(train_dir, "FAKE"), exist_ok=True)
            os.makedirs(os.path.join(train_dir, "REAL"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "FAKE"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "REAL"), exist_ok=True)
            
            # Write dummy blank images
            from PIL import Image as PILImage
            dummy_img = PILImage.new("RGB", (224, 224), (128, 128, 128))
            for i in range(10):
                dummy_img.save(os.path.join(train_dir, "FAKE", f"dummy_{i}.jpg"))
                dummy_img.save(os.path.join(train_dir, "REAL", f"dummy_{i}.jpg"))
                dummy_img.save(os.path.join(test_dir, "FAKE", f"dummy_{i}.jpg"))
                dummy_img.save(os.path.join(test_dir, "REAL", f"dummy_{i}.jpg"))
                
            train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
            test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])
        else:
            sys.exit(1)
            
    # Apply subset if quick run
    if args.quick:
        # Sample 20 train, 10 test images
        train_indices = list(range(0, len(train_dataset), max(1, len(train_dataset) // 20)))[:20]
        test_indices = list(range(0, len(test_dataset), max(1, len(test_dataset) // 10)))[:10]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    elif args.subset_fraction < 1.0:
        # Sample representative fraction
        random.seed(42)
        train_size = max(1, int(len(train_dataset) * args.subset_fraction))
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = Subset(train_dataset, train_indices)
        
        test_size = max(1, int(len(test_dataset) * args.subset_fraction))
        test_indices = random.sample(range(len(test_dataset)), test_size)
        test_dataset = Subset(test_dataset, test_indices)
        logging.info(f"Dataset subsetted to fraction {args.subset_fraction}. Train: {train_size}, Test: {test_size}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    logging.info(f"Dataset mounted. Train size: {dataset_sizes['train']}, Test size: {dataset_sizes['test']}")
    
    # Initialize ConvNeXt-Tiny model
    logging.info("Initializing ConvNeXt-Tiny classifier...")
    try:
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=1)
    except Exception as e:
        logging.warning(f"Could not load pretrained ConvNeXt-Tiny: {e}. Creating scratch model.")
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
        
    model = model.to(device)

    # Freeze backbone weights if requested
    if args.freeze_backbone:
        logging.info("Freezing ConvNeXt-Tiny backbone parameters (training head only)")
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    
    # Precompute features if backbone is frozen to accelerate training on CPU
    if args.freeze_backbone:
        logging.info("Pre-computing features for train and test datasets to accelerate training on CPU...")
        model.eval()
        cached_features = {}
        for phase, loader in [("train", train_loader), ("test", test_loader)]:
            phase_features = []
            phase_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(loader, desc=f"Caching {phase} features"):
                    inputs = inputs.to(device)
                    features = model.forward_features(inputs)
                    pooled = model.forward_head(features, pre_logits=True)
                    phase_features.append(pooled.cpu())
                    phase_labels.append(labels.clone())
            cached_features[phase] = (torch.cat(phase_features, dim=0), torch.cat(phase_labels, dim=0))
            
        from torch.utils.data import TensorDataset
        train_cached_loader = DataLoader(TensorDataset(cached_features["train"][0], cached_features["train"][1]), batch_size=batch_size, shuffle=True)
        test_cached_loader = DataLoader(TensorDataset(cached_features["test"][0], cached_features["test"][1]), batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision components
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Tensorboard logger
    writer = SummaryWriter(log_dir=tensorboard_dir)
    early_stopping = EarlyStopping(patience=5)
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    
    # Training Loop
    logging.info("Beginning model training pipeline...")
    for epoch in range(1, epochs + 1):
        logging.info(f"Epoch {epoch}/{epochs}")
        logging.info("-" * 30)
        
        for phase in ["train", "test"]:
            if phase == "train" and not args.freeze_backbone:
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Select appropriate loader
            if args.freeze_backbone:
                active_loader = train_cached_loader if phase == "train" else test_cached_loader
            else:
                active_loader = train_loader if phase == "train" else test_loader
                
            for inputs, labels in tqdm(active_loader, desc=f"{phase.capitalize()} Batch"):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    # Use autocast for mixed precision
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        if args.freeze_backbone:
                            # Forward pass only through the head classifier
                            outputs = model.head.fc(inputs)
                        else:
                            outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    preds = (outputs >= 0.0).float()
                    
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == "train" and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            
            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            
            # TensorBoard logging
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)
            
            if phase == "test":
                early_stopping(epoch_loss)
                
                # Save epoch checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss
                }
                torch.save(checkpoint, checkpoint_save_path)
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_save_path)
                    logging.info(f"--> Saved improved model to {model_save_path}")
                    
        print()
        if early_stopping.early_stop:
            logging.info("Early stopping triggered. Concluding training.")
            break
            
    writer.close()
    
    # Load best model weights for evaluation
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # Final Scientific Evaluation Metrics
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Class 1 is REAL, Class 0 is FAKE. Probability of being FAKE is 1.0 - prob(REAL)
            # outputs represent logit for REAL
            probs_real = torch.sigmoid(outputs).cpu().squeeze(1).numpy()
            probs_fake = 1.0 - probs_real
            
            preds_real = (outputs >= 0.0).float().cpu().squeeze(1).numpy()
            preds_fake = 1.0 - preds_real
            
            # In validation files, 0 = FAKE, 1 = REAL.
            # We want model evaluations to reflect probability of AI generation (FAKE)
            all_labels.extend(labels.numpy())
            # For metrics, we compare labels (0=FAKE, 1=REAL) with preds_real
            all_preds.extend(preds_real)
            # The calibration/prob checks are usually in terms of FAKE detection
            all_probs.extend(probs_fake)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute performance metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate False Positive Rate (FPR) and False Negative Rate (FNR)
    # Class 0 = FAKE, Class 1 = REAL.
    # False Positive (FPR): REAL classified as FAKE (labeled 1, predicted 0)
    # False Negative (FNR): FAKE classified as REAL (labeled 0, predicted 1)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # Confusion matrix structure:
    # [[TN_fake, FP_real], [FN_fake, TP_real]] -> [[tp_fake, fn_fake], [fp_fake, tn_fake]]
    # Let's compute specifically:
    # Labels: 0 (FAKE), 1 (REAL)
    # Predicted: 0 (FAKE), 1 (REAL)
    # cm[0][0] = TN (actual FAKE, pred FAKE)
    # cm[0][1] = FP (actual FAKE, pred REAL) -> FNR (miss rate)
    # cm[1][0] = FN (actual REAL, pred FAKE) -> FPR (false alarm)
    # cm[1][1] = TP (actual REAL, pred REAL)
    
    # Compute rates
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
        
    fpr_roc, tpr_roc, thresholds = roc_curve(all_labels, all_probs)
    
    roc_list = []
    # Sample points from ROC curve for the frontend chart
    step = max(1, len(fpr_roc) // 20)
    for idx in range(0, len(fpr_roc), step):
        roc_list.append({
            "fpr": float(round(fpr_roc[idx], 4)),
            "tpr": float(round(tpr_roc[idx], 4)),
            "threshold": float(round(thresholds[idx], 4)) if thresholds[idx] <= 1.0 else 1.0
        })
    # Ensure end point is included
    if len(fpr_roc) > 0:
        roc_list.append({
            "fpr": float(round(fpr_roc[-1], 4)),
            "tpr": float(round(tpr_roc[-1], 4)),
            "threshold": 0.0
        })
        
    # ECE and Calibration Bins
    # We calibrate on FAKE class probability. FAKE labels are 1 - all_labels (so FAKE=1, REAL=0)
    fake_labels = 1.0 - all_labels
    ece, bin_stats, histogram_stats = calculate_ece_and_bins(fake_labels, all_probs, n_bins=10)
    brier_score = float(np.mean((all_probs - fake_labels) ** 2))
    
    # Save evaluation report JSON
    eval_data = {
        "metrics": {
            "accuracy": float(round(acc, 4)),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4)),
            "roc_auc": float(round(auc, 4)),
            "fpr": float(round(fpr, 4)),
            "fnr": float(round(fnr, 4))
        },
        "roc_curve": roc_list,
        "probability_distribution": {
            "real": [float(p) for p in all_probs[all_labels == 1]],
            "ai": [float(p) for p in all_probs[all_labels == 0]]
        },
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    
    with open(eval_json_path, "w") as f:
        json.dump(eval_data, f, indent=4)
        
    # Save model calibration parameters JSON
    calib_data = {
        "ece": float(round(ece, 5)),
        "brier_score": float(round(brier_score, 5)),
        "bin_stats": bin_stats,
        "histogram_stats": histogram_stats,
        "ai_temperature": 1.0 # Default unscaled temperature
    }
    
    with open(calib_json_path, "w") as f:
        json.dump(calib_data, f, indent=4)
        
    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Loss Curves")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["test_acc"], label="Test Acc")
    plt.title("Accuracy Curves")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(v2_dir, "loss_accuracy_v2.png"))
    plt.close()
    
    # Save ROC Curve plot
    plt.figure(figsize=(5, 5))
    plt.plot(fpr_roc, tpr_roc, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(v2_dir, "roc_curve_v2.png"))
    plt.close()
    
    logging.info("Training pipeline completed and files written successfully.")
    logging.info(f"Evaluation metrics written to {eval_json_path}")
    logging.info(f"Calibration data written to {calib_json_path}")

if __name__ == "__main__":
    main()
