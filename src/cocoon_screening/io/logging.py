import logging
from pathlib import Path

def setup_logging(
        log_path: str | Path,
        force: bool = False,
        log_level: int = logging.INFO) -> None:
    logger = logging.getLogger("src.cocoon_screening")
    log_file_path = Path(log_path)
    logger.setLevel(log_level)

    if logger.handlers and not force:
        return
    if force:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    log_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_handler = logging.FileHandler(
        log_file_path,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)