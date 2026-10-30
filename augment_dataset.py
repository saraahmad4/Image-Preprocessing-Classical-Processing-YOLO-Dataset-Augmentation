
import cv2
import numpy as np
import random
import shutil
from pathlib import Path

from classical_processing import SOURCE_DIR, read_yolo_label

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "dataset"

IMG_SIZE = 640
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_dataset():

    image_files = sorted(SOURCE_DIR.glob("*.png"))
    samples = []
    missing_labels = []

    for image_path in image_files:

        label_path = SOURCE_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path.name)
            continue

        boxes = read_yolo_label(label_path)

        samples.append({
            "name": image_path.stem,
            "image": image,
            "boxes": boxes
        })

    print("\nOriginal dataset loading:")
    print("Images found:", len(image_files))
    print("Successfully loaded:", len(samples))
    print("Missing labels:", len(missing_labels))

    if missing_labels:
        print("Images without labels:", missing_labels)

    return samples

def letterbox_image_and_boxes(image, boxes, target_size=IMG_SIZE):

    original_h, original_w = image.shape[:2]

    scale = min(
        target_size / original_w,
        target_size / original_h
    )

    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    scale_x = new_w / original_w
    scale_y = new_h / original_h

    updated_boxes = []

    for box in boxes:

        x_center = box["x_center"] * original_w
        y_center = box["y_center"] * original_h

        box_w = box["width"] * original_w
        box_h = box["height"] * original_h

        new_x_center = x_center * scale_x + left
        new_y_center = y_center * scale_y + top

        new_box_w = box_w * scale_x
        new_box_h = box_h * scale_y

        updated_boxes.append({
            "class_id": box["class_id"],
            "x_center": new_x_center / target_size,
            "y_center": new_y_center / target_size,
            "width": new_box_w / target_size,
            "height": new_box_h / target_size
        })

    return letterboxed, updated_boxes

def save_yolo_labels(boxes, output_path):

    with open(output_path, "w") as file:

        for box in boxes:

            file.write(
                f"{box['class_id']} "
                f"{box['x_center']:.6f} "
                f"{box['y_center']:.6f} "
                f"{box['width']:.6f} "
                f"{box['height']:.6f}\n"
            )

def adjust_brightness_contrast(image):

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )


def adjust_hsv(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(np.float32)

    hsv[:, :, 0] += random.uniform(-10, 10)
    hsv[:, :, 1] += random.uniform(-25, 25)
    hsv[:, :, 2] += random.uniform(-25, 25)

    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def add_gaussian_noise(image):

    noise = np.random.normal(
        0,
        random.uniform(5, 15),
        image.shape
    )

    augmented = image.astype(np.float32) + noise

    return np.clip(
        augmented, 0, 255
    ).astype(np.uint8)


def add_salt_pepper_noise(image):

    augmented = image.copy()

    amount = random.uniform(0.001, 0.005)
    total_pixels = image.shape[0] * image.shape[1]

    number_of_pixels = int(amount * total_pixels)

    # Salt pixels
    salt_y = np.random.randint(
        0, image.shape[0], number_of_pixels
    )
    salt_x = np.random.randint(
        0, image.shape[1], number_of_pixels
    )

    augmented[salt_y, salt_x] = 255

    # Pepper pixels
    pepper_y = np.random.randint(
        0, image.shape[0], number_of_pixels
    )
    pepper_x = np.random.randint(
        0, image.shape[1], number_of_pixels
    )

    augmented[pepper_y, pepper_x] = 0

    return augmented


def apply_blur(image):

    if random.choice([True, False]):

        return cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

    return cv2.medianBlur(image, 5)


def apply_photometric_augmentations(image, noise_type):

    augmented = adjust_brightness_contrast(image)
    augmented = adjust_hsv(augmented)

    if noise_type == "gaussian":
        augmented = add_gaussian_noise(augmented)

    elif noise_type == "salt_pepper":
        augmented = add_salt_pepper_noise(augmented)

    augmented = apply_blur(augmented)

    return augmented

def transform_boxes(boxes, matrix, image_width, image_height):

    updated_boxes = []

    for box in boxes:

        x_center = box["x_center"] * image_width
        y_center = box["y_center"] * image_height

        box_w = box["width"] * image_width
        box_h = box["height"] * image_height

        corners = np.array([
            [x_center - box_w / 2, y_center - box_h / 2],
            [x_center + box_w / 2, y_center - box_h / 2],
            [x_center + box_w / 2, y_center + box_h / 2],
            [x_center - box_w / 2, y_center + box_h / 2]
        ], dtype=np.float32)

        transformed_corners = cv2.transform(
            corners.reshape(1, -1, 2),
            matrix
        ).reshape(-1, 2)

        x_min = np.min(transformed_corners[:, 0])
        y_min = np.min(transformed_corners[:, 1])
        x_max = np.max(transformed_corners[:, 0])
        y_max = np.max(transformed_corners[:, 1])

        x_min = np.clip(x_min, 0, image_width)
        y_min = np.clip(y_min, 0, image_height)
        x_max = np.clip(x_max, 0, image_width)
        y_max = np.clip(y_max, 0, image_height)

        new_w = x_max - x_min
        new_h = y_max - y_min

        if new_w < 1 or new_h < 1:
            continue

        updated_boxes.append({
            "class_id": box["class_id"],
            "x_center": ((x_min + x_max) / 2) / image_width,
            "y_center": ((y_min + y_max) / 2) / image_height,
            "width": new_w / image_width,
            "height": new_h / image_height
        })

    return updated_boxes

def horizontal_flip(image, boxes):

    height, width = image.shape[:2]

    flipped = cv2.flip(image, 1)

    matrix = np.array([
        [-1, 0, width - 1],
        [0, 1, 0]
    ], dtype=np.float32)

    updated_boxes = transform_boxes(
        boxes, matrix, width, height
    )

    return flipped, updated_boxes


def vertical_flip(image, boxes):

    height, width = image.shape[:2]

    flipped = cv2.flip(image, 0)

    matrix = np.array([
        [1, 0, 0],
        [0, -1, height - 1]
    ], dtype=np.float32)

    updated_boxes = transform_boxes(
        boxes, matrix, width, height
    )

    return flipped, updated_boxes


def rotate_90(image, boxes):

    height, width = image.shape[:2]

    rotated = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

    matrix = np.array([
        [0, -1, height - 1],
        [1, 0, 0]
    ], dtype=np.float32)

    updated_boxes = transform_boxes(
        boxes, matrix, height, width
    )

    return rotated, updated_boxes


def random_affine(image, boxes):

    height, width = image.shape[:2]

    scale = random.uniform(0.85, 1.15)

    tx = random.uniform(-0.08, 0.08) * width
    ty = random.uniform(-0.08, 0.08) * height

    center_x = width / 2
    center_y = height / 2

    matrix = np.array([
        [scale, 0, (1 - scale) * center_x + tx],
        [0, scale, (1 - scale) * center_y + ty]
    ], dtype=np.float32)

    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)
    )

    updated_boxes = transform_boxes(
        boxes, matrix, width, height
    )

    return transformed, updated_boxes

def generate_augmentations(image, boxes):

    variants = []

    # Variant 1: Photometric only
    variants.append((
        "photo",
        apply_photometric_augmentations(image, "gaussian"),
        boxes.copy()
    ))

    # Variant 2: Horizontal flip + photometric
    flipped, updated_boxes = horizontal_flip(image, boxes)

    variants.append((
        "hflip",
        apply_photometric_augmentations(flipped, "salt_pepper"),
        updated_boxes
    ))

    # Variant 3: Vertical flip + photometric
    flipped, updated_boxes = vertical_flip(image, boxes)

    variants.append((
        "vflip",
        apply_photometric_augmentations(flipped, "gaussian"),
        updated_boxes
    ))

    # Variant 4: 90-degree rotation + photometric
    rotated, updated_boxes = rotate_90(image, boxes)

    variants.append((
        "rot90",
        apply_photometric_augmentations(rotated, "salt_pepper"),
        updated_boxes
    ))

    # Variant 5: Random affine + photometric
    transformed, updated_boxes = random_affine(image, boxes)

    variants.append((
        "affine",
        apply_photometric_augmentations(transformed, "gaussian"),
        updated_boxes
    ))

    return variants

def split_dataset(groups):

    groups = groups.copy()

    random.Random(RANDOM_SEED).shuffle(groups)

    total = len(groups)

    train_count = int(total * 0.70)
    val_count = int(total * 0.20)

    splits = {
        "train": groups[:train_count],
        "val": groups[train_count:train_count + val_count],
        "test": groups[train_count + val_count:]
    }

    print("\nOriginal image groups split:")

    for split_name, split_groups in splits.items():

        print(
            f"{split_name}: {len(split_groups)} groups, "
            f"{len(split_groups) * 6} images"
        )

    return splits

def save_dataset(splits):

    if OUTPUT_DIR.resolve() == SOURCE_DIR.resolve():
        raise RuntimeError("Output and source directories must differ!")

    # Recreate generated output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name, groups in splits.items():

        image_dir = OUTPUT_DIR / split_name / "images"
        label_dir = OUTPUT_DIR / split_name / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for group in groups:

            # Save original
            original_name = group["name"]

            original_image_path = image_dir / f"{original_name}.png"
            original_label_path = label_dir / f"{original_name}.txt"

            success = cv2.imwrite(
                str(original_image_path),
                group["image"]
            )

            if not success:
                raise RuntimeError(
                    f"Could not save {original_image_path}"
                )

            save_yolo_labels(
                group["boxes"],
                original_label_path
            )

            saved_count += 1

            # Save five augmented variants
            for suffix, augmented_image, updated_boxes in group["variants"]:

                new_name = f"{original_name}_{suffix}"

                image_path = image_dir / f"{new_name}.png"
                label_path = label_dir / f"{new_name}.txt"

                success = cv2.imwrite(
                    str(image_path),
                    augmented_image
                )

                if not success:
                    raise RuntimeError(
                        f"Could not save {image_path}"
                    )

                save_yolo_labels(
                    updated_boxes,
                    label_path
                )

                saved_count += 1

        print(f"{split_name}: saved {saved_count} images")


def create_data_yaml():

    yaml_content = """path: .
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: car
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w") as file:
        file.write(yaml_content)

    print("\ndata.yaml created:", yaml_path)

def main():

    print("=" * 55)
    print("YOLO DATASET PREPROCESSING AND AUGMENTATION")
    print("=" * 55)

    # Load the 50 original images
    samples = load_dataset()

    if len(samples) != 50:
        raise ValueError(
            f"Expected 50 original images, found {len(samples)}"
        )

    groups = []

    print("\nResizing and generating augmentations...")

    for index, sample in enumerate(samples, start=1):

        # Read original boxes
        boxes = sample["boxes"]

        # Resize original image and update its boxes
        resized_image, resized_boxes = letterbox_image_and_boxes(
            sample["image"],
            boxes,
            IMG_SIZE
        )

        # Generate five augmented variants
        variants = generate_augmentations(
            resized_image,
            resized_boxes
        )

        groups.append({
            "name": sample["name"],
            "image": resized_image,
            "boxes": resized_boxes,
            "variants": variants
        })

        if index % 10 == 0:
            print(f"Processed {index}/{len(samples)} originals")

    # Split groups only AFTER augmentation
    splits = split_dataset(groups)

    # Save all images and labels
    save_dataset(splits)

    # Generate YOLO configuration
    create_data_yaml()

    print("\n" + "=" * 55)
    print("DATASET GENERATION COMPLETED!")
    print("Original images: 50")
    print("Augmented images: 250")
    print("Total images: 300")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 55)


if __name__ == "__main__":
    main()
