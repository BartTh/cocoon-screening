from pathlib import Path
import json
import logging
import os
import sys
from datetime import datetime

APP_NAME = "CocoonScreening"

logger = logging.getLogger(__name__)

def bundled_resource_path(relative_path: str | Path) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parents[2] / relative_path

def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    if not base.is_absolute():
        base = (Path.home() / "AppData" / "Local" if sys.platform == "win32"
                else Path.home() / ".local" / "share")
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

def user_log_path() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"app_{datetime.now():%Y%m%d_%H%M%S}.log"

def app_version() -> str:
    """Version stamped into the bundle at build time.

    The build workflow writes version.txt (the git tag for a tagged build, otherwise the
    short commit SHA) and PyInstaller bundles it, so a DMG can always be traced back to
    the commit it came from. Returns "dev" when running from a checkout.
    """
    try:
        version = bundled_resource_path("version.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return "dev"
    return version or "dev"


def bundled_models_dir() -> Path:
    return bundled_resource_path("models")

def bundled_configs_dir() -> Path:
    return bundled_resource_path("configs")

def app_install_dir() -> Path:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if sys.platform == "darwin":
            for parent in executable.parents:
                if parent.suffix.lower() == ".app":
                    return parent.parent
        return executable.parent
    return Path(__file__).resolve().parents[2]


def ui_state_path() -> Path:
    """Small file remembering what the user last chose, next to the logs."""
    return user_data_dir() / "ui_state.json"


def remembered_directory(key: str) -> Path | None:
    """Folder last used for this purpose, or None.

    Browsing used to start from the home folder on every launch, because the fallback
    chain ended at app_install_dir()/data, which does not exist beside an installed .app.
    """
    try:
        state = json.loads(ui_state_path().read_text(encoding="utf-8"))
        value = state.get("directories", {}).get(key)
    except (OSError, ValueError):
        return None
    if not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def remember_directory(key: str, directory: str | Path) -> None:
    """Record a chosen folder. Never let a cache failure break the UI."""
    try:
        path = Path(directory)
        if not path.is_dir():
            path = path.parent
        try:
            state = json.loads(ui_state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        state.setdefault("directories", {})[key] = str(path.resolve())
        ui_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as error:
        logger.info("Could not remember the %s directory: %s", key, error)


def normalize_directory(value: str | Path) -> Path:
    """Require an explicit location, independent of the launch working directory."""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    if not text:
        raise ValueError("Please choose a directory using Browse or enter its full path.")
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        raise ValueError("Please enter a full directory path or select it using Browse.")
    return path.resolve()


def browse_start_directory(*candidates) -> Path:
    """Ignore missing/unavailable defaults without creating any dataset folders."""
    for candidate in (*candidates, Path.home()):
        if candidate is None:
            continue
        try:
            path = normalize_directory(candidate)
            if path.is_dir() and os.access(path, os.R_OK | os.X_OK):
                return path
        except (OSError, ValueError, RuntimeError):
            continue
    return Path.home()


def default_images_dir() -> Path:
    return browse_start_directory(app_install_dir() / "data" / "images")


def default_raw_dir() -> Path:
    return browse_start_directory(app_install_dir() / "data" / "raw")


def default_review_dir() -> Path:
    return browse_start_directory(app_install_dir() / "data" / "for_review")


def review_dataset_root(review_dir: str | Path) -> Path:
    """The folder holding images/ and labels/ that a review folder belongs to.

    Review mode reads the original image and label map, and writes decisions, there. It
    used to be exactly review_dir.parent, which forced every review folder to sit directly
    beside images/ and labels/. Now the nearest such folder up to four levels above counts,
    so review folders can be grouped (interim/review/<name>, interim/review/done/<name>).
    Nearest wins: a self-contained round with its own images/ and labels/ keeps using them.
    Falls back to review_dir.parent, the old behaviour, when none is found.
    """
    review_dir = Path(review_dir)
    for candidate in [review_dir.parent, *review_dir.parent.parents][:4]:
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    return review_dir.parent
