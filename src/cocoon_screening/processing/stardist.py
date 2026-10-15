from pathlib import Path
import cv2
import numpy as np
import logging
import math
from src.cocoon_screening.paths import bundled_models_dir
from src.cocoon_screening.io.atomic import image_bytes, write_bytes

logger = logging.getLogger(__name__)

# The network has a U-Net of depth 3 on a grid of 2, so every input axis must be a
# multiple of 2**3 * 2 = 16; otherwise the skip connections fail to concatenate.
DIVISOR = 16
GRID = (2, 2)
# StarDist's defaults. The model folder has no thresholds.json, so nothing else supplies them.
PROB_THRESH = 0.5
NMS_THRESH = 0.4


class StardistPredictor:
    """Runs the custom StarDist model through onnxruntime.

    model.onnx is exported from the Keras weights by tools/export_stardist_onnx.py, which
    is the only place TensorFlow is still needed.
    """

    def __init__(self):
        self._session = None

    def session(self):
        if self._session is None:
            import onnxruntime as ort
            path = bundled_models_dir() / "bee_cocoon_stardist" / "model.onnx"
            options = ort.SessionOptions()
            # A whole tray runs untiled; the arena would keep ~400 MB more at peak and hold
            # on to it between trays. Costs well under a second per tray.
            options.enable_cpu_mem_arena = False
            self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        return self._session

    def predict(self, image_path: Path):
        from csbdeep.utils import normalize
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32)
        return self._segment(normalize(image, 1, 99.8))

    def _segment(self, image: np.ndarray):
        from stardist.nms import non_maximum_suppression
        from stardist.geometry import dist_to_coord, polygons_to_label

        height, width = image.shape
        pad = [(0, math.ceil(n / DIVISOR) * DIVISOR - n) for n in (height, width)]
        padded = np.pad(image, pad, mode="reflect")[np.newaxis, ..., np.newaxis]

        session = self.session()
        prob, dist = session.run(None, {session.get_inputs()[0].name: padded.astype(np.float32)})
        # Outputs are at grid resolution; crop away what the padding added.
        rows, cols = (math.ceil(n / g) for n, g in zip((height, width), GRID))
        prob = prob[0, :rows, :cols, 0]
        dist = dist[0, :rows, :cols, :]

        points, probi, disti = non_maximum_suppression(
            dist, prob, grid=GRID, prob_thresh=PROB_THRESH, nms_thresh=NMS_THRESH)
        labels = polygons_to_label(disti, points, prob=probi, shape=(height, width))
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
