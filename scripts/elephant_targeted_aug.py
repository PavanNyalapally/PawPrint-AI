"""
Targeted augmentation for elephant footprint images.

Usage:
    python3 scripts/elephant_targeted_aug.py <num_to_add>

Notes:
- Only ORIGINAL images are augmented (never aug_*, targeted_aug_*)
- Safe to run multiple times
- Designed for footprint realism
"""

import sys
import os
import random
import uuid
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

RNG = random.Random(123)

BASE_TRAIN = os.path.join("dataset", "classification", "train", "elephant")
BASE_RAW = os.path.join("dataset", "raw", "elephant")

IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def random_crop_resize(img):
    """Random crop with resize, preserving footprint."""
    w, h = img.size
    scale = RNG.uniform(0.7, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    left = RNG.randint(0, max(0, w - new_w))
    top = RNG.randint(0, max(0, h - new_h))

    img = img.crop((left, top, left + new_w, top + new_h))
    return img.resize(IMAGE_SIZE, Image.BILINEAR)


def color_jitter(img):
    if RNG.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(RNG.uniform(0.8, 1.2))
    if RNG.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(RNG.uniform(0.85, 1.2))
    return img


def maybe_blur(img):
    if RNG.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(RNG.uniform(0.5, 2.0)))
    return img


def maybe_cutout(img):
    """Neutral cutout (dark patch), footprint-safe."""
    if RNG.random() < 0.25:
        w, h = img.size
        cw = int(RNG.uniform(0.08, 0.2) * w)
        ch = int(RNG.uniform(0.08, 0.2) * h)
        x = RNG.randint(0, w - cw)
        y = RNG.randint(0, h - ch)

        patch = Image.new("RGB", (cw, ch), (0, 0, 0))
        img.paste(patch, (x, y))
    return img


def augment_image(path):
    img = Image.open(path).convert("RGB")

    img = random_crop_resize(img)

    # Small rotation with expand to avoid clipping
    if RNG.random() < 0.5:
        img = img.rotate(RNG.uniform(-12, 12), expand=True, fillcolor=(0, 0, 0))
        img = img.resize(IMAGE_SIZE, Image.BILINEAR)

    # Horizontal flip
    if RNG.random() < 0.5:
        img = ImageOps.mirror(img)

    img = color_jitter(img)
    img = maybe_blur(img)
    img = maybe_cutout(img)

    return img


def gather_original_sources():
    """Gather only ORIGINAL images (no augmented ones)."""
    sources = []

    for base in [BASE_RAW, BASE_TRAIN]:
        if not os.path.isdir(base):
            continue

        for f in os.listdir(base):
            if (
                f.lower().endswith(VALID_EXTENSIONS)
                and not f.startswith(("aug_", "targeted_aug_"))
            ):
                sources.append(os.path.join(base, f))

    if not sources:
        raise SystemExit("❌ No original elephant images found")

    return sources


def main(add_count):
    os.makedirs(BASE_TRAIN, exist_ok=True)

    sources = gather_original_sources()

    existing = [
        f for f in os.listdir(BASE_TRAIN)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    print(f"📊 Existing elephant train images: {len(existing)}")
    print(f"➕ Targeted augmentations to add: {add_count}")

    for i in range(add_count):
        src = RNG.choice(sources)
        aug = augment_image(src)

        unique_id = uuid.uuid4().hex[:8]
        fname = f"targeted_aug_{unique_id}.jpg"
        dst = os.path.join(BASE_TRAIN, fname)

        aug.save(dst, quality=90)

        if (i + 1) % 20 == 0:
            print(f"🔄 Created {i + 1}/{add_count} images")

    print(f"✅ Finished: added {add_count} elephant images to {BASE_TRAIN}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/elephant_targeted_aug.py <num_to_add>")
        raise SystemExit(1)

    count = int(sys.argv[1])
    main(count)
