from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def draw_instance_overlay(image, mask, color=(0, 255, 0), alpha=0.35):
    if image.ndim == 2:
        base = cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    elif image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    else:
        base = image.copy()

    overlay = base.copy()

    overlay[mask] = color

    result = cv2.addWeighted(
        overlay, alpha,
        base, 1 - alpha,
        0
    )

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(result, contours, -1, color, 1)

    return result

def render_completed_image_overlay(image, crops, classifications, labels_set, alpha=0.15):
    if image.ndim == 2:
        base = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    elif image.shape[2] == 4:
        base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    else:
        base = image.copy()

    overlay = base.copy()

    for instance_id, crop in crops.items():
        matches = classifications.loc[
            classifications["instance_id"] == instance_id,
            "class_index",
        ]

        if matches.empty:
            continue

        class_index = matches.iloc[0]

        if pd.isna(class_index):
            continue

        class_index = int(class_index)
        color = hex_to_bgr(labels_set.color_for_index(class_index))

        x1, y1, x2, y2 = crop["bounds"]
        mask = crop["mask"]

        overlay_region = overlay[y1:y2 + 1, x1:x2 + 1]
        overlay_region[mask] = color

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        shifted_contours = [
            contour + np.array([[[x1, y1]]])
            for contour in contours
        ]

        cv2.drawContours(
            overlay,
            shifted_contours,
            -1,
            color,
            1,
        )

    return cv2.addWeighted(
        overlay,
        alpha,
        base,
        1 - alpha,
        0,
    )

def crop_instances(image, label_image, padding=0):
    boxes = get_instance_bounding_boxes(label_image)
    height, width = image.shape[:2]
    crops = {}

    for box in boxes:
        min_x, min_y, max_x, max_y = box["bounds"]

        x1 = max(0, min_x - padding)
        y1 = max(0, min_y - padding)
        x2 = min(width - 1, max_x + padding)
        y2 = min(height - 1, max_y + padding)

        crop = image[y1:y2 + 1, x1:x2 + 1].copy()

        mask = (
            label_image[y1:y2 + 1, x1:x2 + 1] == box["instance_id"]
        )

        crops[box["instance_id"]] = {
            "instance_id": box["instance_id"],
            "image": crop,
            "mask": mask,
            "bounds": (x1, y1, x2, y2),
            "class_index": np.nan
        }

    return crops

def get_instance_bounding_boxes(label_image):
    if label_image.ndim == 3:
        label_image = label_image[:, :, 0]

    # Group foreground coordinates once instead of scanning the full image
    # for every instance. Sparse IDs and disconnected parts remain supported.
    ys, xs = np.nonzero(label_image)
    if not len(xs):
        return []
    ids, groups = np.unique(label_image[ys, xs], return_inverse=True)
    min_x = np.full(len(ids), label_image.shape[1], dtype=np.intp)
    min_y = np.full(len(ids), label_image.shape[0], dtype=np.intp)
    max_x = np.zeros(len(ids), dtype=np.intp)
    max_y = np.zeros(len(ids), dtype=np.intp)
    np.minimum.at(min_x, groups, xs)
    np.minimum.at(min_y, groups, ys)
    np.maximum.at(max_x, groups, xs)
    np.maximum.at(max_y, groups, ys)
    return [
        {"instance_id": int(instance_id),
         "bounds": (int(min_x[i]), int(min_y[i]), int(max_x[i]), int(max_y[i]))}
        for i, instance_id in enumerate(ids)
    ]

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 8:
        hex_color = hex_color[:6]

    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    return (blue, green, red)
