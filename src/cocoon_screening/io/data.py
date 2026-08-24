from pathlib import Path
import pandas as pd
import logging

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_LABEL_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp"}
logger = logging.getLogger(__name__)

def load_images_from_directory(images_directory):
    images = {}

    for image_path in sorted(Path(images_directory).iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images[image_path.stem] = image_path

    return images

def load_labels_from_directory(labels_directory):
    labels = {}
    labels_directory = Path(labels_directory)

    if not labels_directory.exists():
        logger.info(f"Labels directory '{labels_directory}' does not yet exist. Labels need to be created.")
        return labels

    for label_path in sorted(labels_directory.iterdir()):
        if label_path.is_file() and label_path.suffix.lower() in SUPPORTED_LABEL_EXTENSIONS:
            labels[label_path.stem] = label_path

    return labels

def image_registry_path(images_directory: Path) -> Path:
    return images_directory.parent / "image_registry.csv"

def get_batch_id(images_directory: Path, image_id: str) -> str:
    path_to_registry = image_registry_path(images_directory)
    if not path_to_registry.exists():
        logger.warning(f"Warning: Image registry file '{path_to_registry}' does not exist.")
        return ""
    image_registry = pd.read_csv(path_to_registry)

    image_row = image_registry.loc[image_registry["image_id"] == image_id, "batch_id"]
    if image_row.empty:
        logger.warning(f"Warning: Image_id '{image_id}' not found in image image registry.")
        return ""
    return image_row.iloc[0]