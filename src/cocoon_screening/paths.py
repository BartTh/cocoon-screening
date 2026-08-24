from pathlib import Path
import os
import sys
from datetime import datetime

APP_NAME = "CocoonScreening"

def bundled_resource_path(relative_path: str | Path) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parents[2] / relative_path

def user_data_dir() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", Path.home()))
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def user_log_path() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"

def bundled_models_dir() -> Path:
    return bundled_resource_path("models")

def bundled_configs_dir() -> Path:
    return bundled_resource_path("configs")

def app_install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def default_images_dir() -> Path:
    local_images = app_install_dir() / "data" / "images"

    if local_images.exists():
        return local_images

    fallback_directory = Path.home()
    return fallback_directory

def default_raw_dir() -> Path:
    local_raw = app_install_dir() / "data" / "raw"

    if local_raw.exists():
        return local_raw

    fallback_directory = Path.home()
    return fallback_directory