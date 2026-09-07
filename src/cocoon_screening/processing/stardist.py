from pathlib import Path
import cv2
import numpy as np
import logging
from src.cocoon_screening.paths import bundled_models_dir
from src.cocoon_screening.io.atomic import image_bytes, write_bytes

logger = logging.getLogger(__name__)

class StardistPredictor:
    def __init__(self, use_custom_model: bool = True):
        self.use_custom_model = use_custom_model
        self._model = None

    def model(self):
        if self._model is None:
            from stardist.models import StarDist2D
            if self.use_custom_model:
                self._model = StarDist2D(
                    None,
                    name="bee_cocoon_stardist",
                    basedir=str(bundled_models_dir()),
                )
            else:
                self._model = StarDist2D.from_pretrained("2D_versatile_fluo")

        return self._model

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
        image_normalized = normalize(image, 1, 99.8)
        labels, details = self.model().predict_instances(
            image_normalized,
            n_tiles=(4, 4),
            show_tile_progress=False,
            verbose=False
        )

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
