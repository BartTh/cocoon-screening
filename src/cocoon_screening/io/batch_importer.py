from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import yaml
import csv
import pydicom
import numpy as np
import cv2

IMAGE_ID_PATTERN = re.compile(r"^XR_(\d{6})$")
logger = logging.getLogger(__name__)

@dataclass
class BatchImportResult:
    batch_id: str
    discovered: int
    imported: int
    skipped_duplicates: int
    failed: int
    output_directory: Path

def append_to_image_registry(image_registry_path: Path, rows: list[dict]) -> None:
    if not rows:
        logger.info("No new registry rows to append.")
        return
    
    image_registry_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = image_registry_path.exists() and image_registry_path.stat().st_size > 0

    with image_registry_path.open("a", encoding="utf-8", newline="",) as image_registry_file:
        writer = csv.DictWriter(image_registry_file, fieldnames=
                                [
                                    "batch_id",
                                    "image_id",
                                    "initial_dicom_name",
                                ])
        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)
    logger.info("Appended %s rows to image registry: %s", len(rows), image_registry_path)

def adjust_dicom_exposure(pixels):
    pixels = pixels - np.min(pixels)
    pixels = (pixels / np.max(pixels) * 255).astype(np.uint8)
    return pixels

def load_existing_image_ids(image_registry_path: Path) -> set[str]:
    if not image_registry_path.exists():
        return set()

    existing_ids = set()

    with image_registry_path.open("r", encoding="utf-8", newline="") as image_registry_file:
        reader = csv.DictReader(image_registry_file)
        for row in reader:
            existing_ids.add(row["image_id"])

    return existing_ids

def import_batch(
    batch_directory: Path,
    output_directory: Path,
    image_registry_path: Path,
    batch_id: str,
    output_format: str = "jpg",
) -> BatchImportResult:
    """
    Import and preprocess a batch of DICOM X-ray images with continuous ID assignment.
    """
    logger.info(
        "Starting DICOM batch import | batch_id=%s batch_directory=%s output_directory=%s",
        batch_id,
        batch_directory,
        output_directory,
    )

    dicom_files = [
        path
        for path in batch_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {"", ".dcm"}
    ]
    logger.info("Discovered %s DICOM files in batch %s.", len(dicom_files), batch_id)

    output_directory.mkdir(parents=True, exist_ok=True)

    existing_image_ids = load_existing_image_ids(image_registry_path)
    new_image_registry_rows = []
    skipped_duplicates = 0
    failed = 0

    next_id = next_image_id(existing_image_ids)
    logger.info("Next available image ID: XR_%06d", next_id)

    for dicom_file in dicom_files:
        new_image_id = f"XR_{next_id:06d}"

        if new_image_id in existing_image_ids:
            logger.warning(
                "Duplicate image ID %s found. Skipping DICOM file: %s",
                new_image_id,
                dicom_file,
            )
            skipped_duplicates += 1
            next_id += 1
            continue

        try:
            logger.info("Importing DICOM file %s as %s", dicom_file, new_image_id)
            dicom_data = pydicom.dcmread(dicom_file)
            pixels = dicom_data.pixel_array
            pixels = adjust_dicom_exposure(pixels)

            output_file = output_directory / f"{new_image_id}.{output_format}"
            if not cv2.imwrite(str(output_file), pixels):
                raise IOError(f"Could not write output image: {output_file}")

            existing_image_ids.add(new_image_id)
            new_image_registry_rows.append(
                {
                    "batch_id": batch_id,
                    "image_id": new_image_id,
                    "initial_dicom_name": dicom_file.name,
                }
            )
            next_id += 1

        except Exception:
            failed += 1
            logger.exception("Failed to import DICOM file: %s", dicom_file)

    append_to_image_registry(image_registry_path, new_image_registry_rows)
    logger.info(
        "Finished DICOM batch import | batch_id=%s discovered=%s imported=%s skipped_duplicates=%s failed=%s",
        batch_id,
        len(dicom_files),
        len(new_image_registry_rows),
        skipped_duplicates,
        failed,
    )

    return BatchImportResult(
        batch_id=batch_id,
        discovered=len(dicom_files),
        imported=len(new_image_registry_rows),
        skipped_duplicates=skipped_duplicates,
        failed=failed,
        output_directory=output_directory,
    )
        
def next_image_id(existing_ids: set[str]) -> int:
    numbers = []
    for image_id in existing_ids:
        match = IMAGE_ID_PATTERN.match(image_id)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1
