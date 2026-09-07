from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ReviewItem:
    crop_path: Path
    image_id: str
    instance_id: int
    bounds: tuple[int, int, int, int]


def parse_review_filename(path: Path) -> ReviewItem:
    """
    <image_id>_instance_<id>_x<x1>_y<y1>-x<x2>_y<y2>.<suffix>
    """

    filename_pattern = re.compile(
        r"^(?P<image_id>.+)"
        r"_instance_(?P<instance_id>\d+)"
        r"_x(?P<x1>\d+)_y(?P<y1>\d+)"
        r"-x(?P<x2>\d+)_y(?P<y2>\d+)"
        r"\.(?P<suffix>png|jpg|jpeg|tif|tiff|bmp)$",
        re.IGNORECASE,
    )

    match = filename_pattern.match(path.name)
    if not match:
        logger.error(f"Invalid filename format: {path.name}")
        raise ValueError(f"Invalid filename format: {path.name}")

    if (int(match['instance_id']) <= 0 or int(match['x2']) < int(match['x1'])
            or int(match['y2']) < int(match['y1'])):
        raise ValueError(f"Invalid instance or bounds in {path.name}")

    return ReviewItem(
        crop_path=path,
        image_id=match.group("image_id"),
        instance_id=int(match.group("instance_id")),
        bounds=
        (
            int(match.group("x1")),
            int(match.group("y1")),
            int(match.group("x2")),
            int(match.group("y2")),
        ),
    )
