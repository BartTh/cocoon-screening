from pathlib import Path
import cv2
import numpy as np
import logging
import math
from src.cocoon_screening.paths import bundled_models_dir
from src.cocoon_screening.io.atomic import image_bytes, write_bytes
from src.cocoon_screening.processing.stardist_post import (
    non_maximum_suppression, normalize, polygons_to_label, dist_to_coord)

logger = logging.getLogger(__name__)

# The network has a U-Net of depth 3 on a grid of 2, so every input axis must be a
# multiple of 2**3 * 2 = 16; otherwise the skip connections fail to concatenate.
DIVISOR = 16
GRID = (2, 2)
# StarDist's defaults. The model folder has no thresholds.json, so nothing else supplies them.
PROB_THRESH = 0.5
NMS_THRESH = 0.4

# Tiling. A pixel influences outputs up to ~95 px away (measured), so each tile carries
# TILE_MARGIN px of context beyond the part it contributes; that part then sees exactly the
# input it would see untiled. Tiles start on multiples of DIVISOR so the network's
# downsampling grid lines up with the untiled one. TILE_SIZE bounds the part a tile
# contributes. Measured on a 2653x2156 tray, network peak RSS / time: untiled 1.56 GB / 5.3 s,
# 1024 px 0.98 GB / 5.2 s, 512 px 0.56 GB / 6.9 s, 256 px 0.42 GB / 11.4 s.
TILE_MARGIN = 128
TILE_SIZE = 512


class StardistPredictor:
    """Runs the custom StarDist model through onnxruntime.

    model.onnx is exported from the Keras weights by tools/export_stardist_onnx.py, which
    is the only place TensorFlow is still needed.
    """

    def __init__(self, tile_size: int | None = TILE_SIZE):
        self.tile_size = tile_size
        self._session = None

    def session(self):
        if self._session is None:
            import onnxruntime as ort
            path = bundled_models_dir() / "bee_cocoon_stardist" / "model.onnx"
            options = ort.SessionOptions()
            # The arena would keep its peak allocation between trays.
            options.enable_cpu_mem_arena = False
            self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        return self._session

    def predict(self, image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return self._segment(normalize(image.astype(np.float32), 1, 99.8))

    def network(self, image: np.ndarray):
        """prob (H/2, W/2) and dist (H/2, W/2, n_rays) for a normalised image."""
        height, width = image.shape
        # Reflect-pad at the end to a multiple of DIVISOR, as StarDist does.
        padded = np.pad(image, [(0, -height % DIVISOR), (0, -width % DIVISOR)], mode="reflect")
        rows, cols = (math.ceil(n / g) for n, g in zip((height, width), GRID))
        if self.tile_size is None:
            prob, dist = self._run(padded)
            return prob[:rows, :cols], dist[:rows, :cols]

        prob = dist = None
        for r0, r1 in _tile_spans(padded.shape[0], self.tile_size):
            for c0, c1 in _tile_spans(padded.shape[1], self.tile_size):
                t0, t1 = max(0, r0 - TILE_MARGIN), min(padded.shape[0], r1 + TILE_MARGIN)
                s0, s1 = max(0, c0 - TILE_MARGIN), min(padded.shape[1], c1 + TILE_MARGIN)
                tile_prob, tile_dist = self._run(padded[t0:t1, s0:s1])
                if prob is None:
                    shape = (padded.shape[0] // GRID[0], padded.shape[1] // GRID[1])
                    prob = np.empty(shape, tile_prob.dtype)
                    dist = np.empty(shape + tile_dist.shape[-1:], tile_dist.dtype)
                gr, gc = GRID
                core = (slice((r0 - t0) // gr, (r1 - t0) // gr), slice((c0 - s0) // gc, (c1 - s0) // gc))
                prob[r0 // gr:r1 // gr, c0 // gc:c1 // gc] = tile_prob[core]
                dist[r0 // gr:r1 // gr, c0 // gc:c1 // gc] = tile_dist[core]
        return prob[:rows, :cols], dist[:rows, :cols]

    def _run(self, padded: np.ndarray):
        session = self.session()
        prob, dist = session.run(None, {session.get_inputs()[0].name: padded[np.newaxis, ..., np.newaxis]})
        return prob[0, ..., 0], dist[0]

    def _segment(self, image: np.ndarray):
        prob, dist = self.network(image)
        points, probi, disti = non_maximum_suppression(
            dist, prob, grid=GRID, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH)
        labels = polygons_to_label(disti, points, probi, image.shape)
        details = dict(coord=dist_to_coord(disti, points), points=points, prob=probi)
        return labels, details

    def create_label_map(
            self,
            image_path: Path,
            label_path: Path,
    ) -> Path:
        logger.info(
            "No label map found under %s for image %s. Running StarDist prediction...",
            label_path,
            image_path.name,
        )
        labels, details = self.predict(image_path)

        label_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if labels.max() > np.iinfo(np.uint16).max:
            raise ValueError("Too many instances for a 16-bit label map.")
        write_bytes(label_path, image_bytes(labels.astype(np.uint16), label_path))
        logger.info("Saved label map under %s with %s detected cocoon instances", label_path, int(labels.max()))
        return label_path


def _tile_spans(length: int, tile_size: int):
    """Split [0, length) into near-equal spans whose bounds are multiples of DIVISOR."""
    count = max(1, math.ceil(length / tile_size))
    step = math.ceil(length / count / DIVISOR) * DIVISOR
    return [(start, min(start + step, length)) for start in range(0, length, step)]
