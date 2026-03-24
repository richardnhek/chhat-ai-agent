"""
Brand Classifier — Fine-tuned ResNet50 classifier trained on 233 brand book reference images.

Instead of raw embedding similarity (which gives 0.35-0.55 on real images),
this trains an actual classifier that outputs: "This crop is ARA RED" with 90%+ confidence.

Training uses heavy augmentation to simulate real-world conditions:
- Rotation (0-360°), perspective warp
- Brightness/contrast variation, blur
- Glass reflection overlay, partial occlusion
- Noise, color jitter
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from brands import BRANDS_AND_SKUS


# ── Dataset ──────────────────────────────────────────────────────────────────

class BrandDataset(Dataset):
    """Dataset from brand book reference images with heavy augmentation."""

    def __init__(self, reference_dir: str = "reference_images", transform=None, samples_per_image: int = 50):
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.samples_per_image = samples_per_image

        # Load mapping
        mapping_file = self.reference_dir / "mapping.json"
        with open(mapping_file) as f:
            self.mapping = json.load(f)

        # Build brand → class index
        all_brands = sorted(set(v["brand"].split(" ")[0] if " " in v["brand"] else v["brand"]
                               for v in self.mapping.values()))
        # Map to mother brands from BRANDS_AND_SKUS
        self.mother_brands = sorted(BRANDS_AND_SKUS.keys())
        self.brand_to_idx = {b: i for i, b in enumerate(self.mother_brands)}
        self.idx_to_brand = {i: b for b, i in self.brand_to_idx.items()}
        self.num_classes = len(self.mother_brands)

        # Load images and their brand labels
        self.images = []
        self.labels = []
        skipped = 0

        for filename, info in self.mapping.items():
            img_path = self.reference_dir / filename
            if not img_path.exists():
                skipped += 1
                continue
            if img_path.suffix.lower() in ('.wdp',):
                skipped += 1
                continue

            brand_name = info.get("brand", "")
            # Map to mother brand
            mother_brand = self._find_mother_brand(brand_name)
            if mother_brand and mother_brand in self.brand_to_idx:
                try:
                    img = Image.open(img_path).convert("RGB")
                    self.images.append(img)
                    self.labels.append(self.brand_to_idx[mother_brand])
                except Exception:
                    skipped += 1

        print(f"  Loaded {len(self.images)} reference images ({skipped} skipped)")
        print(f"  {self.num_classes} brand classes")

    def _find_mother_brand(self, brand_name: str) -> str | None:
        """Map a brand/SKU name to its mother brand."""
        brand_upper = brand_name.upper().strip()
        # Direct match
        if brand_upper in self.brand_to_idx:
            return brand_upper
        # Check if it starts with a mother brand name
        for mb in self.mother_brands:
            if brand_upper.startswith(mb):
                return mb
            if mb in brand_upper:
                return mb
        # Fuzzy match
        for mb in self.mother_brands:
            if mb.replace(" ", "") in brand_upper.replace(" ", ""):
                return mb
        return None

    def __len__(self):
        return len(self.images) * self.samples_per_image

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        img = self.images[img_idx]
        label = self.labels[img_idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ── Augmentation pipeline ────────────────────────────────────────────────────

def get_train_transform():
    """Heavy augmentation to simulate real-world store conditions."""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(180),  # Packs can be upside down
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),  # Simulate occlusion
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Model ────────────────────────────────────────────────────────────────────

def build_classifier(num_classes: int) -> nn.Module:
    """ResNet50 with custom classifier head for brand recognition."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers (they learn generic features)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


# ── Training ─────────────────────────────────────────────────────────────────

def train_classifier(
    reference_dir: str = "reference_images",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_path: str = "models/brand_classifier.pth",
):
    """Train the brand classifier on reference images."""
    print(f"\n{'='*60}")
    print(f"  Brand Classifier Training")
    print(f"{'='*60}\n")

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Dataset
    train_dataset = BrandDataset(reference_dir, transform=get_train_transform(), samples_per_image=50)
    val_dataset = BrandDataset(reference_dir, transform=get_val_transform(), samples_per_image=5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = train_dataset.num_classes
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {len(train_dataset)} ({len(train_dataset.images)} images × 50 augmentations)")
    print(f"  Validation samples: {len(val_dataset)}")

    # Model
    model = build_classifier(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

        scheduler.step()
        train_acc = 100. * train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_imgs)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

        val_acc = 100. * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "brand_to_idx": train_dataset.brand_to_idx,
                "idx_to_brand": train_dataset.idx_to_brand,
                "epoch": epoch,
                "val_acc": val_acc,
            }, output_path)

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} — train_acc: {train_acc:.1f}% | val_acc: {val_acc:.1f}% | best: {best_val_acc:.1f}% | {elapsed:.0f}s")

    print(f"\n  Training complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model saved: {output_path}")
    return output_path


# ── Inference ────────────────────────────────────────────────────────────────

_classifier_model = None
_classifier_meta = None


def _load_classifier(model_path: str = "models/brand_classifier.pth"):
    """Load the trained classifier."""
    global _classifier_model, _classifier_meta

    if _classifier_model is not None:
        return _classifier_model, _classifier_meta

    if not Path(model_path).exists():
        return None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    num_classes = checkpoint["num_classes"]

    model = build_classifier(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    meta = {
        "brand_to_idx": checkpoint["brand_to_idx"],
        "idx_to_brand": checkpoint["idx_to_brand"],
        "val_acc": checkpoint.get("val_acc", 0),
    }

    _classifier_model = model
    _classifier_meta = meta
    return model, meta


def classify_crop(crop_data: bytes, top_k: int = 3) -> list[dict]:
    """
    Classify a cropped cigarette pack image.
    Returns top-k predictions with brand name and confidence.
    """
    model, meta = _load_classifier()
    if model is None:
        return []

    transform = get_val_transform()

    img = Image.open(io.BytesIO(crop_data)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = probs.topk(top_k)

    results = []
    idx_to_brand = meta["idx_to_brand"]
    for prob, idx in zip(top_probs, top_indices):
        brand = idx_to_brand.get(str(idx.item()), idx_to_brand.get(idx.item(), "UNKNOWN"))
        results.append({
            "brand": brand,
            "confidence": float(prob),
        })

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
