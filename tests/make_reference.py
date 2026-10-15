#!/usr/bin/env python3
"""Freeze the predictor's current label maps, to test a later change against.

    python tests/make_reference.py <images dir> <out dir>
    COCOON_IMAGES_DIR=<images dir> COCOON_REFERENCE_DIR=<out dir> pytest tests

Writes <out dir>/XR_*.npz with labels, points and prob per tray. Run it before the change.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cocoon_screening.processing.stardist import StardistPredictor  # noqa: E402


def main():
    images, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    predictor = StardistPredictor()
    for path in sorted(images.glob("XR_*.jpg")):
        start = time.perf_counter()
        labels, details = predictor.predict(path)
        np.savez_compressed(out / f"{path.stem}.npz", labels=labels.astype(np.uint16),
                            points=details["points"], prob=details["prob"])
        print(f"{path.stem}: {int(labels.max())} instances, {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
