
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
import shutil

# ==============================
# 1. DIRECTORY CONFIGURATION
# ==============================
PROJECT_DIR = Path.cwd()
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = PROJECT_DIR / "dataset"
SOURCE_DIR = PROJECT_DIR / "senna-dataset"

# ==============================
# 2. LOAD IMAGES AND LABELS
# ==============================
image_files = sorted(SOURCE_DIR.glob("*.png"))
print("Number of images found:", len(image_files))
dataset = []
missing_labels = []

for image_path in image_files:

    label_path = SOURCE_DIR / (image_path.stem + ".txt")

    if not label_path.exists():
        missing_labels.append(image_path.name)
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path.name)
        continue

    height, width = image.shape[:2]

    dataset.append({
        "name": image_path.stem,
        "image": image,
        "label_path": label_path,
        "width": width,
        "height": height
    })

print("Successfully loaded:", len(dataset))
print("Missing labels:", len(missing_labels))

if missing_labels:
    print("Images without labels:", missing_labels)

# ==============================
# 3. INSPECT DATASET
# ==============================
if dataset:
    print("\nFirst image information:")

    sample = dataset[0]

    print("Filename:", sample["name"])
    print("Image dimensions:",
            sample["width"], "x", sample["height"])

    print("\nCorresponding YOLO annotation:")

    with open(sample["label_path"], "r") as file:
        print(file.read())

else:
    print("No valid image-label pairs found!")

# ==============================
# 4. YOLO LABEL READING
# ==============================
def read_yolo_label(label_path):

    boxes = []

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

            boxes.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height
            })

    return boxes


# ==============================
# 5. YOLO TO PIXEL COORDINATES
# ==============================
def yolo_to_pixels(box, image_width, image_height):

    x_center = box["x_center"] * image_width
    y_center = box["y_center"] * image_height

    box_width = box["width"] * image_width
    box_height = box["height"] * image_height

    x_min = int(round(x_center - box_width / 2))
    y_min = int(round(y_center - box_height / 2))

    x_max = int(round(x_center + box_width / 2))
    y_max = int(round(y_center + box_height / 2))

    return x_min, y_min, x_max, y_max


# ==============================
# 6. DRAW GROUND-TRUTH BOXES
# ==============================
def draw_yolo_box(image, label_path):

    result = image.copy()

    height, width = image.shape[:2]

    boxes = read_yolo_label(label_path)

    for box in boxes:

        x_min, y_min, x_max, y_max = yolo_to_pixels(
            box, width, height
        )

        # Draw green bounding box
        cv2.rectangle(
            result,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            3
        )

        # Draw class ID
        cv2.putText(
            result,
            f"Class {box['class_id']}",
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result


# ==============================
# 7. VISUALIZE ANNOTATIONS
# ==============================
sample_count = min(5, len(dataset))

fig, axes = plt.subplots(
    1, sample_count,
    figsize=(20, 5)
)

if sample_count == 1:
    axes = [axes]

for i in range(sample_count):

    sample = dataset[i]

    annotated = draw_yolo_box(
        sample["image"],
        sample["label_path"]
    )

    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    axes[i].imshow(annotated)
    axes[i].set_title(sample["name"])
    axes[i].axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "ground_truth_visualization.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==============================
# 8. CLASSICAL IMAGE PROCESSING
# ==============================
def classical_processing(image):

    # 1. Convert BGR to Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 2. Convert BGR to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # 3. Gaussian Blur
    gaussian = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # 4. Median Blur
    median = cv2.medianBlur(
        image,
        5
    )

    # 5. Bilateral Filter
    bilateral = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    # Return all processed images
    return {
        "gray": gray,
        "hsv": hsv,
        "gaussian": gaussian,
        "median": median,
        "bilateral": bilateral
    }


# ==============================
# 9. DISPLAY PROCESSING RESULTS
# ==============================
sample = dataset[0]
image = sample["image"]
processed = classical_processing(image)

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 9)
)

axes[0, 0].imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
axes[0, 0].set_title("Original Image")

# Grayscale
axes[0, 1].imshow(
    processed["gray"],
    cmap="gray"
)
axes[0, 1].set_title("Grayscale")

# HSV
axes[0, 2].imshow(
    cv2.cvtColor(
        processed["hsv"],
        cv2.COLOR_HSV2RGB
    )
)
axes[0, 2].set_title("HSV")

# Gaussian Blur
axes[1, 0].imshow(
    cv2.cvtColor(
        processed["gaussian"],
        cv2.COLOR_BGR2RGB
    )
)
axes[1, 0].set_title("Gaussian Blur")

# Median Blur
axes[1, 1].imshow(
    cv2.cvtColor(
        processed["median"],
        cv2.COLOR_BGR2RGB
    )
)
axes[1, 1].set_title("Median Blur")

# Bilateral Filter
axes[1, 2].imshow(
    cv2.cvtColor(
        processed["bilateral"],
        cv2.COLOR_BGR2RGB
    )
)
axes[1, 2].set_title("Bilateral Filter")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "filtering_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


def apply_thresholding(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slightly blur the image to reduce noise
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    otsu_value, otsu_binary = cv2.threshold(
        blurred_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Inverted Otsu result
    otsu_inverse = cv2.bitwise_not(otsu_binary)

    # Adaptive thresholding
    adaptive_binary = cv2.adaptiveThreshold(
        blurred_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # Inverted adaptive result
    adaptive_inverse = cv2.bitwise_not(adaptive_binary)

    return {
        "gray": gray,
        "otsu": otsu_binary,
        "otsu_inverse": otsu_inverse,
        "adaptive": adaptive_binary,
        "adaptive_inverse": adaptive_inverse,
        "otsu_value": otsu_value
    }


# Test on the first image
sample = dataset[0]
image = sample["image"]

thresholds = apply_thresholding(image)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

images_to_show = [
    (image, "Original Image"),
    (thresholds["gray"], "Grayscale"),
    (thresholds["otsu"], "Otsu Threshold"),
    (thresholds["otsu_inverse"], "Otsu Inverted"),
    (thresholds["adaptive"], "Adaptive Threshold"),
    (thresholds["adaptive_inverse"], "Adaptive Inverted")
]

for ax, (img, title) in zip(axes.flat, images_to_show):
    ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(
    f"Thresholding Comparison - Otsu Value: {thresholds['otsu_value']:.0f}",
    fontsize=14
)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "thresholding_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


def apply_morphology(binary_image):
    # Create a 5x5 kernel
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological operations
    erosion = cv2.erode(binary_image, kernel, iterations=1)

    dilation = cv2.dilate(binary_image, kernel, iterations=1)

    opening = cv2.morphologyEx(
        binary_image, cv2.MORPH_OPEN, kernel
    )

    closing = cv2.morphologyEx(
        binary_image, cv2.MORPH_CLOSE, kernel
    )

    return {
        "original": binary_image,
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing
    }


# Use Otsu's result
otsu = thresholds["otsu"]

morphology = apply_morphology(otsu)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

operations = [
    ("original", "Original Otsu"),
    ("erosion", "Erosion"),
    ("dilation", "Dilation"),
    ("opening", "Opening"),
    ("closing", "Closing")
]

for ax, (key, title) in zip(axes.flat, operations):
    ax.imshow(morphology[key], cmap="gray")
    ax.set_title(title)
    ax.axis("off")

axes[1, 2].axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "morphology_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


def detect_edges_and_contours(image, label_path):
    height, width = image.shape[:2]

    # 1. Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Get the Otsu binary image
    _, otsu = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Clean the binary image using opening and closing
    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(
        otsu, cv2.MORPH_OPEN, kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, kernel
    )

    # 5. Read the YOLO bounding box
    boxes = read_yolo_label(label_path)

    if not boxes:
        return None

    # Use the first annotation
    gt_box = yolo_to_pixels(boxes[0], width, height)
    x_min, y_min, x_max, y_max = gt_box

    # Keep the ROI inside the image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    # 6. Crop the binary image around the car
    roi = cleaned[y_min:y_max, x_min:x_max]

    # 7. Find contours
    contours, _ = cv2.findContours(
        roi.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contours found for", Path(label_path).name)
        return None

    # Select the largest contour in the ROI
    largest_contour = max(contours, key=cv2.contourArea)

    # Move contour coordinates back to the full image
    largest_contour[:, 0, 0] += x_min
    largest_contour[:, 0, 1] += y_min

    # 8. Calculate the minimum-area rotated rectangle
    rotated_rect = cv2.minAreaRect(largest_contour)
    rectangle_points = cv2.boxPoints(rotated_rect)
    rectangle_points = np.int32(rectangle_points)

    # 9. Get an axis-aligned bounding box around the contour
    cx, cy, cw, ch = cv2.boundingRect(largest_contour)

    # 10. Calculate IoU with the YOLO bounding box
    intersection_x1 = max(cx, gt_box[0])
    intersection_y1 = max(cy, gt_box[1])
    intersection_x2 = min(cx + cw, gt_box[2])
    intersection_y2 = min(cy + ch, gt_box[3])

    intersection = (
        max(0, intersection_x2 - intersection_x1)
        * max(0, intersection_y2 - intersection_y1)
    )

    contour_area_box = cw * ch
    gt_area = (
        (gt_box[2] - gt_box[0])
        * (gt_box[3] - gt_box[1])
    )

    union = contour_area_box + gt_area - intersection
    iou = intersection / union if union > 0 else 0

    # 11. Draw results
    result = image.copy()

    # Draw the detected contour in red
    cv2.drawContours(
        result, [largest_contour], -1, (0, 0, 255), 3
    )

    # Draw the rotated rectangle in blue
    cv2.drawContours(
        result, [rectangle_points], 0, (255, 0, 0), 3
    )

    # Draw the YOLO ground-truth box in green
    cv2.rectangle(
        result,
        (gt_box[0], gt_box[1]),
        (gt_box[2], gt_box[3]),
        (0, 255, 0),
        3
    )

    return {
        "edges": edges,
        "cleaned": cleaned,
        "contour": result,
        "iou": iou,
        "rotated_rect": rotated_rect,
        "gt_box": gt_box
    }


# Test on the first image
sample = dataset[0]

results = detect_edges_and_contours(
    sample["image"],
    sample["label_path"]
)

if results is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(results["edges"], cmap="gray")
    axes[0].set_title("Canny Edge Detection")

    axes[1].imshow(results["cleaned"], cmap="gray")
    axes[1].set_title("Cleaned Otsu Mask")

    axes[2].imshow(
        cv2.cvtColor(results["contour"], cv2.COLOR_BGR2RGB)
    )
    axes[2].set_title(
        f"Contour + Bounding Boxes (IoU: {results['iou']:.3f})"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "canny_contour_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("Contour vs YOLO IoU:", round(results["iou"], 3))
    print("Rotated rectangle:", results["rotated_rect"])


def create_classical_comparison(sample):
    image = sample["image"]
    label_path = sample["label_path"]

    # Ground-truth visualization
    annotated = draw_yolo_box(image, label_path)

    # Classical processing
    processed = classical_processing(image)
    thresholds = apply_thresholding(image)
    morphology = apply_morphology(thresholds["otsu"])

    # Edge and contour detection
    results = detect_edges_and_contours(image, label_path)

    # Create comparison grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    images = [
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Original + YOLO"),
        (processed["gray"], "Grayscale"),
        (cv2.cvtColor(processed["gaussian"], cv2.COLOR_BGR2RGB), "Gaussian Blur"),
        (cv2.cvtColor(processed["bilateral"], cv2.COLOR_BGR2RGB), "Bilateral Filter"),
        (thresholds["otsu"], "Otsu Threshold"),
        (thresholds["otsu_inverse"], "Otsu Inverted"),
        (morphology["opening"], "Morphological Opening"),
        (morphology["closing"], "Morphological Closing"),
    ]

    for ax, (img, title) in zip(axes.flat, images):
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    if results is not None:
        axes[2, 0].imshow(results["edges"], cmap="gray")
        axes[2, 0].set_title("Canny Edges")

        axes[2, 1].imshow(results["cleaned"], cmap="gray")
        axes[2, 1].set_title("Cleaned Binary Mask")

        axes[2, 2].imshow(
            cv2.cvtColor(results["contour"], cv2.COLOR_BGR2RGB)
        )
        axes[2, 2].set_title(
            f"Contour + Boxes (IoU: {results['iou']:.3f})"
        )

    axes[2, 3].axis("off")

    plt.suptitle(f"Classical Processing: {sample['name']}", fontsize=16)
    plt.tight_layout()

    output_path = RESULTS_DIR / "5 scenes" / f"{sample['name']}_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results["iou"] if results is not None else None


# Process five images
ious = []

for sample in dataset[:5]:
    iou = create_classical_comparison(sample)
    ious.append(iou)
    print(sample["name"], "- IoU:", iou)

print("\nAll comparison grids saved in:", RESULTS_DIR)