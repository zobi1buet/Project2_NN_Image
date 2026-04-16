"""Shared utilities for the Veggie Classification project."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def get_device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def show_images(images, labels, class_names, predictions=None, n=8):
    """Display a grid of images with labels and optional predictions.

    Args:
        images: Tensor (N, C, H, W) in [0, 1] range.
        labels: List/tensor of true label indices.
        class_names: List of class name strings.
        predictions: Optional list/tensor of predicted label indices.
        n: Number of images to show.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis("off")

        true_label = class_names[labels[i]]
        if predictions is not None:
            pred_label = class_names[predictions[i]]
            color = "green" if predictions[i] == labels[i] else "red"
            axes[i].set_title(f"{pred_label}\n({true_label})", fontsize=8, color=color)
        else:
            axes[i].set_title(true_label, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader. Returns (loss, accuracy %)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def per_class_accuracy(model, dataloader, class_names, device):
    """Print per-class accuracy breakdown."""
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()

    print("Per-class accuracy:")
    print("-" * 35)
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]:20s}: {acc:.1f}%")
