"""Annotation I/O jobs. No widgets or mutable window state belong here."""
from dataclasses import dataclass
from pathlib import Path
from src.cocoon_screening.paths import normalize_directory
import cv2
import pandas as pd
from src.cocoon_screening.io.atomic import recover_commit, write_bytes, csv_bytes, image_bytes
from src.cocoon_screening.io.data import get_batch_id, load_images_from_directory, load_labels_from_directory
from src.cocoon_screening.domain.classifications import validate_classifications, classification_progress
from src.cocoon_screening.processing.image_processing import crop_instances, render_completed_image_overlay


@dataclass
class AnnotationResult:
    payload: object = None
    saved: object = None
    error: Exception = None
    overlay_failed: bool = False


def prepare_annotation_data(image_path, label_paths, predictor, label_set):
    recover_commit(image_path.parent.parent / f".{image_path.stem}.exclusion.pending.json")
    label_path = label_paths.get(image_path.stem)
    if label_path is None:
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.png"
        predictor.create_label_map(image_path, label_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if image is None or labels is None:
        raise ValueError(f"Could not read image or label map for {image_path.name}")
    if labels.ndim == 3:
        labels = labels[:, :, 0]
    if image.shape[:2] != labels.shape[:2]:
        raise ValueError(f"Image and label sizes differ for {image_path.name}")
    crops = crop_instances(image, labels, padding=10)
    csv_path = image_path.parent.parent / "labels" / f"{image_path.stem}.csv"
    if csv_path.exists():
        classifications = validate_classifications(pd.read_csv(csv_path), len(label_set))
        if set(classifications["instance_id"]) - set(crops):
            raise ValueError(f"Classification CSV contains IDs absent from {label_path.name}")
    else:
        classifications = pd.DataFrame({
            "instance_id": list(crops), "class_index": pd.NA,
        })
        classifications = validate_classifications(classifications, len(label_set))
    batch_id = get_batch_id(image_path.parent, image_path.stem)
    pixmaps = {}
    if not csv_path.exists():
        write_bytes(csv_path, csv_bytes(classifications))
    return dict(crops=crops, classifications=classifications,
                label_path=label_path, batch_id=batch_id, pixmaps=pixmaps)


def run_annotation_job(snapshot, label_set, load=None):
    """Retain the successful CSV save even if a later operation fails."""
    result = AnnotationResult()
    try:
        if snapshot is not None:
            path, table, crops = snapshot
            journal = path.parent.parent / f".{path.stem}.exclusion.pending.json"
            # Reload is the recovery path for an interrupted exclusion.
            if not journal.exists():
                table = validate_classifications(table, len(label_set))
                csv_path = path.parent.parent / "labels" / f"{path.stem}.csv"
                write_bytes(csv_path, csv_bytes(table))
                result.saved = (path, table)
                if classification_progress(table, crops, len(label_set))[1]:
                    try:
                        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        if image is None:
                            raise ValueError(f"Could not read image: {path}")
                        overlay = render_completed_image_overlay(image, crops, table, label_set, alpha=0.15)
                        target = path.parent.parent / "overlays" / f"{path.stem}_class_overlay.png"
                        write_bytes(target, image_bytes(overlay, target))
                    except Exception:
                        result.overlay_failed = True
                        raise
        if load is not None:
            result.payload = load()
    except Exception as error:
        result.error = error
    return result


def prepare_directory(directory, predictor, label_set):
    directory = normalize_directory(directory)
    recover_commit(directory.parent / ".import.pending.json")
    images = load_images_from_directory(directory)
    labels = load_labels_from_directory(directory.parent / "labels")
    path = next(iter(images.values()), None)
    prepared = prepare_annotation_data(path, labels, predictor, label_set) if path else None
    return directory, images, labels, path, prepared
