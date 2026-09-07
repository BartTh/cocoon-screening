from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import logging
import re

import cv2
import numpy as np
import pandas as pd
import pydicom

from src.cocoon_screening.io.atomic import commit_files, recover_commit, csv_bytes, image_bytes
from src.cocoon_screening.io.data import SUPPORTED_IMAGE_EXTENSIONS

IMAGE_ID_PATTERN = re.compile(r"^XR_(\d{6,})$")
REGISTRY_COLUMNS = ["batch_id", "image_id", "initial_dicom_name", "sop_instance_uid", "source_sha256"]
logger = logging.getLogger(__name__)


@dataclass
class BatchImportResult:
    batch_id: str
    discovered: int
    imported: int
    skipped_duplicates: int
    failed: int
    output_directory: Path


def read_registry(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"batch_id", "image_id", "initial_dicom_name"}.issubset(table.columns):
        raise ValueError(f"Invalid image registry: {path}")
    if table["image_id"].eq("").any() or table["image_id"].duplicated().any():
        raise ValueError("Registry image IDs must be nonempty and unique.")
    for column in REGISTRY_COLUMNS:
        if column not in table:
            table[column] = ""
    return table


def load_existing_image_ids(image_registry_path):
    return set(read_registry(image_registry_path)["image_id"])


def adjust_dicom_exposure(pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("DICOM pixels must be nonempty and finite.")
    pixels = pixels - pixels.min()
    maximum = pixels.max()
    if maximum == 0:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return (pixels / maximum * 255).astype(np.uint8)


def next_image_id(existing_ids):
    numbers = [int(m.group(1)) for value in existing_ids
               if (m := IMAGE_ID_PATTERN.fullmatch(value))]
    return max(numbers, default=0) + 1


def import_batch(batch_directory, output_directory, image_registry_path,
                 batch_id, output_format="jpg"):
    batch_directory = Path(batch_directory).resolve()
    output_directory = Path(output_directory).resolve()
    image_registry_path = Path(image_registry_path).resolve()
    root = image_registry_path.parent
    if not output_directory.is_relative_to(root):
        raise ValueError("Output images must be within the registry's dataset directory.")
    output_format = output_format.lower().lstrip(".")
    if "." + output_format not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported output format: {output_format}")
    batch_id = batch_id.strip()
    if not batch_id:
        raise ValueError("Batch ID cannot be empty.")
    dicom_files = sorted(path for path in batch_directory.iterdir()
                         if path.is_file() and path.suffix.lower() in {"", ".dcm"})
    output_directory.mkdir(parents=True, exist_ok=True)
    journal = root / ".import.pending.json"
    recover_commit(journal)
    table = read_registry(image_registry_path)
    # Reserve IDs from actual files too, including exports absent from a registry.
    files = {p.stem: p for p in output_directory.iterdir() if p.is_file()}
    used_ids = set(table["image_id"]) | set(files)
    next_id = next_image_id(used_ids)
    known_uids = set(table["sop_instance_uid"]) - {""}
    known_hashes = set(table["source_sha256"]) - {""}
    imported = skipped = failed = 0

    for dicom_file in dicom_files:
        try:
            with dicom_file.open("rb") as stream:
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_hash = digest.hexdigest()
            data = pydicom.dcmread(dicom_file)
            uid = str(getattr(data, "SOPInstanceUID", "")).strip()
            if source_hash in known_hashes or (uid and uid in known_uids):
                skipped += 1
                continue
            pixels = adjust_dicom_exposure(data.pixel_array)
            if pixels.ndim != 2:
                raise ValueError("Only single-frame grayscale X-rays are supported.")

            # Upgrade legacy rows only when the original name/batch AND the
            # encoded image agree. Names alone are not proof of duplication.
            legacy = table.loc[
                table["batch_id"].eq(batch_id)
                & table["initial_dicom_name"].eq(dicom_file.name)
                & table["source_sha256"].eq("")
            ]
            matched = None
            for index, row in legacy.iterrows():
                old_path = files.get(row["image_id"])
                if old_path and old_path.read_bytes() == image_bytes(pixels, old_path):
                    matched = index
                    break
            if matched is not None:
                updated = table.copy()
                updated.loc[matched, ["sop_instance_uid", "source_sha256"]] = [uid, source_hash]
                commit_files(journal, {image_registry_path: csv_bytes(updated)})
                table = updated
                known_hashes.add(source_hash)
                if uid:
                    known_uids.add(uid)
                skipped += 1
                continue

            new_id = f"XR_{next_id:06d}"
            while new_id in used_ids or any(output_directory.glob(new_id + ".*")):
                next_id += 1
                new_id = f"XR_{next_id:06d}"
            output_file = output_directory / f"{new_id}.{output_format}"
            encoded = image_bytes(pixels, output_file)
            updated = pd.concat([table, pd.DataFrame([{
                "batch_id": batch_id, "image_id": new_id,
                "initial_dicom_name": dicom_file.name,
                "sop_instance_uid": uid, "source_sha256": source_hash,
            }])], ignore_index=True)
            # A staged image + registry are recoverable together on the next run.
            commit_files(journal, {
                output_file: encoded, image_registry_path: csv_bytes(updated),
            })
            table = updated
            files[new_id] = output_file
            used_ids.add(new_id)
            known_hashes.add(source_hash)
            if uid:
                known_uids.add(uid)
            next_id += 1
            imported += 1
        except Exception:
            if journal.exists():
                # Stop before another file can overwrite an interrupted commit.
                raise RuntimeError(
                    "Import interrupted while saving. Retry this import to recover "
                    "the staged image and registry."
                )
            failed += 1
            logger.exception("Failed to import DICOM file: %s", dicom_file)

    logger.info("Import %s: %s imported, %s duplicates, %s failed",
                batch_id, imported, skipped, failed)
    return BatchImportResult(batch_id, len(dicom_files), imported, skipped,
                             failed, output_directory)
