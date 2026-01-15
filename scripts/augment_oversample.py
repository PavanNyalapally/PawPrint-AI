"""
Simple augmentation-based oversampler for small classes.

Usage:
    python3 scripts/augment_oversample.py <class_name> <target_count>

Example:
    python3 scripts/augment_oversample.py deer 150

Notes:
- Only ORIGINAL images are augmented (never aug_*.jpg)
- Safe to run multiple times
- Designed for footprint images (no aggressive transforms)
"""

import sys
import os
import random
from PIL import Image, ImageEnhance

RNG = random.Random(42)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def augment_image(img_path):
    """Apply light, realistic augmentations."""
    img = Image.open(img_path).convert('RGB')

    # Random rotation (keep full footprint)
    angle = RNG.uniform(-20, 20)
    img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0))

    # Random horizontal flip
    if RNG.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(RNG.uniform(0.8, 1.2))

    # Random contrast
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(RNG.uniform(0.85, 1.15))

    return img


def oversample_class(base_dir, klass, target):
    train_dir = os.path.join(base_dir, 'train', klass)

    if not os.path.isdir(train_dir):
        raise SystemExit(f"❌ Train dir not found: {train_dir}")

    # Only ORIGINAL images (exclude aug_*)
    original_files = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
        and not f.startswith("aug_")
    ]

    if not original_files:
        raise SystemExit(f"❌ No original images found for class: {klass}")

    existing_files = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    n_current = len(existing_files)
    print(f"📊 {klass}: current={n_current}, target={target}")

    if n_current >= target:
        print(f"✅ No oversampling needed for {klass}")
        return

    i = 0
    while n_current < target:
        src_name = RNG.choice(original_files)
        src_path = os.path.join(train_dir, src_name)

        img = augment_image(src_path)

        new_name = f"aug_{i}_{src_name}"
        dst_path = os.path.join(train_dir, new_name)

        if os.path.exists(dst_path):
            i += 1
            continue

        img.save(dst_path, quality=90)
        i += 1
        n_current += 1

        if i % 10 == 0:
            print(f"🔄 {klass}: generated {i} augmented images (total={n_current})")

    print(f"✅ Finished oversampling {klass}: final count={n_current}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/augment_oversample.py <class_name> <target_count>")
        raise SystemExit(1)

    klass = sys.argv[1]
    target = int(sys.argv[2])

    base_dir = os.path.join("dataset", "classification")
    oversample_class(base_dir, klass, target)
