
import re
import math
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from classical_processing import draw_yolo_box, PROJECT_DIR, OUTPUT_DIR

VERIFY_DIR = PROJECT_DIR / "verification_results"

IMG_SIZE = 640

EXPECTED_COUNTS = {
    "train": 210,
    "val": 60,
    "test": 30
}

AUGMENTATION_TYPES = [
    "photo",
    "hflip",
    "vflip",
    "rot90",
    "affine"
]


def verify_dataset():

    print("=" * 55)
    print("DATASET VERIFICATION")
    print("=" * 55)

    all_valid = True

    for split_name, expected_count in EXPECTED_COUNTS.items():

        image_dir = OUTPUT_DIR / split_name / "images"
        label_dir = OUTPUT_DIR / split_name / "labels"

        image_files = sorted(image_dir.glob("*.png"))
        label_files = sorted(label_dir.glob("*.txt"))

        image_names = {p.stem for p in image_files}
        label_names = {p.stem for p in label_files}

        print(f"\n{split_name.upper()}")

        print("Images:", len(image_files))
        print("Labels:", len(label_files))

        # Check image-label matching
        if image_names != label_names:

            print("ERROR: Image-label mismatch!")
            print("Images without labels:", image_names - label_names)
            print("Labels without images:", label_names - image_names)

            all_valid = False

        # Check expected image count
        if len(image_files) != expected_count:

            print("ERROR: Incorrect image count!")
            all_valid = False

        if len(label_files) != expected_count:

            print("ERROR: Incorrect label count!")
            all_valid = False

        # Check every image and its label
        for image_path in image_files:

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            image = cv2.imread(str(image_path))

            if image is None:

                print("ERROR: Cannot read image:", image_path.name)
                all_valid = False
                continue

            if image.shape[:2] != (IMG_SIZE, IMG_SIZE):

                print("ERROR: Incorrect dimensions:", image_path.name)
                all_valid = False

            # Validate YOLO labels
            with open(label_path, "r") as file:

                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            if not lines:

                print("ERROR: Empty label:", label_path.name)
                all_valid = False
                continue

            for line_number, line in enumerate(lines, start=1):

                values = line.split()

                if len(values) != 5:

                    print("ERROR: Invalid label format:", label_path.name)
                    all_valid = False
                    continue

                try:

                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:])

                except ValueError:

                    print("ERROR: Non-numeric label:", label_path.name)
                    all_valid = False
                    continue

                coordinates = [x, y, w, h]

                if (
                    class_id != 0
                    or not all(math.isfinite(v) for v in coordinates)
                    or not all(0 <= v <= 1 for v in coordinates)
                    or w <= 0
                    or h <= 0
                    or x - w / 2 < -1e-6
                    or y - h / 2 < -1e-6
                    or x + w / 2 > 1 + 1e-6
                    or y + h / 2 > 1 + 1e-6
                ):

                    print(
                        "ERROR: Invalid coordinates or class:",
                        label_path.name,
                        "line",
                        line_number
                    )

                    all_valid = False

    print("\n" + "=" * 55)

    if all_valid:
        print("FILE, DIMENSION AND LABEL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED. Review the errors above.")

    return all_valid



def verify_groups():

    print("\n" + "=" * 55)
    print("AUGMENTATION GROUP VERIFICATION")
    print("=" * 55)

    all_valid = True

    group_splits = {}

    for split_name in EXPECTED_COUNTS:

        image_dir = OUTPUT_DIR / split_name / "images"

        for image_path in image_dir.glob("*.png"):

            # Extract original ID, e.g. img_000 from img_000_hflip
            match = re.match(r"^(img_\d+)", image_path.stem)

            if not match:

                print("ERROR: Unexpected filename:", image_path.name)
                all_valid = False
                continue

            original_id = match.group(1)

            group_splits.setdefault(original_id, []).append(
                (image_path.stem, split_name)
            )

    for original_id, members in sorted(group_splits.items()):

        splits = {split_name for _, split_name in members}

        if len(splits) != 1:

            print("ERROR: Group found in multiple splits:", original_id)
            all_valid = False

        if len(members) != 6:

            print(
                f"ERROR: {original_id} has {len(members)} images instead of 6"
            )
            all_valid = False

        expected_members = {
            original_id,
            *[f"{original_id}_{suffix}" for suffix in AUGMENTATION_TYPES]
        }

        actual_members = {name for name, _ in members}

        if actual_members != expected_members:

            print("ERROR: Missing or unexpected variants:", original_id)
            print("Missing:", expected_members - actual_members)
            print("Unexpected:", actual_members - expected_members)

            all_valid = False

    if len(group_splits) != 50:

        print("ERROR: Expected 50 original image groups!")
        print("Found:", len(group_splits))
        all_valid = False

    if all_valid:
        print("All 50 groups contain exactly 6 images.")
        print("No original group is split across train, val or test.")

    return all_valid



def visualize_samples():

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, split_name in zip(axes, splits):

        image_dir = OUTPUT_DIR / split_name / "images"
        label_dir = OUTPUT_DIR / split_name / "labels"

        image_files = sorted(image_dir.glob("*.png"))

        if not image_files:

            ax.set_title(f"{split_name}: No images")
            ax.axis("off")
            continue

        image_path = image_files[0]
        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        annotated = draw_yolo_box(image, label_path)

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(annotated)
        ax.set_title(f"{split_name.upper()}: {image_path.stem}")
        ax.axis("off")

        cv2.imwrite(
            str(VERIFY_DIR / f"{split_name}_verification.png"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        )

    plt.tight_layout()

    plt.savefig(
        VERIFY_DIR / "all_splits_verification.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


def visualize_augmented_boxes(): # visualize 10 augmented images with bounding boxes

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    image_dir = OUTPUT_DIR / "train" / "images"
    label_dir = OUTPUT_DIR / "train" / "labels"

    selected_images = []

    # Select two images from each of the five augmentation types
    for suffix in AUGMENTATION_TYPES:

        matching_images = sorted(
            image_dir.glob(f"*_{suffix}.png")
        )

        selected_images.extend(matching_images[:2])

    if len(selected_images) < 10:

        print("ERROR: Could not find 10 augmented images.")
        return False

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))

    for ax, image_path in zip(axes.flat, selected_images):

        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None or not label_path.exists():

            print("ERROR: Cannot load:", image_path.name)
            ax.set_title("Loading error")
            ax.axis("off")
            continue

        annotated = draw_yolo_box(image, label_path)

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(annotated)
        ax.set_title(image_path.stem, fontsize=9)
        ax.axis("off")

    plt.suptitle(
        "Bounding Box Verification - 10 Augmented Images",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        VERIFY_DIR / "10_augmented_boxes_verification.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\n10 augmented images displayed.")
    print(
        "Saved to:",
        VERIFY_DIR / "10_augmented_boxes_verification.png"
    )

    return True

if __name__ == "__main__":

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    files_valid = verify_dataset()
    groups_valid = verify_groups()

    if files_valid and groups_valid:

        print("\nAll automatic checks passed!")

        visualize_samples()
        visualize_augmented_boxes()

    else:

        print("\nFix the reported errors before visual verification.")
