"""Scrollable CT view for review crops that come with a CT stack.

A review folder made from a CT scan (04_data/scripts/ct/make_review_sheets.py) puts a
<crop name>.ct.npz beside each review crop:

  slices      uint8 (n, h, w)  CT slices parallel to the tray, already windowed, aligned to
                               the X-ray crop and at its display scale; index 0 = highest
  heights_mm  float (n,)       height of each slice above the tray floor
  xray        uint8 (h, w)     the X-ray crop, contrast-stretched
  outline     uint8 (h, w)     the cocoon's mask in the same frame

Review crops without one are shown exactly as before.
"""

from pathlib import Path

import cv2
import numpy as np

DISPLAY_HEIGHT = 520         # px on screen; the stacks are small (~150 px), so they are enlarged
GAP = 6
OUTLINE_COLOUR = (0, 200, 255)


def stack_path(crop_path: Path) -> Path:
    return crop_path.with_name(crop_path.stem + ".ct.npz")


def load_ct_stack(crop_path: Path):
    """The CT stack beside a review crop, or None."""
    path = stack_path(Path(crop_path))
    if not path.is_file():
        return None
    with np.load(path) as data:
        stack = {key: data[key] for key in ("slices", "heights_mm", "xray", "outline")}
    if stack["slices"].ndim != 3 or stack["slices"].shape[1:] != stack["xray"].shape:
        return None
    return stack


def densest_slice(stack) -> int:
    """Start where the cocoon holds the most material: usually through its contents."""
    mask = stack["outline"] > 0
    if not mask.any():
        return len(stack["slices"]) // 2
    return int(np.argmax([s[mask].mean() for s in stack["slices"]]))


def render_ct_view(stack, index: int, show_outline: bool = True) -> np.ndarray:
    """X-ray | CT slice, side by side, enlarged, with the height of the slice."""
    index = int(np.clip(index, 0, len(stack["slices"]) - 1))
    panels = [("X-ray", stack["xray"]),
              (f"CT {stack['heights_mm'][index]:.1f} mm above floor  ({index + 1}/{len(stack['slices'])})",
               stack["slices"][index])]
    factor = DISPLAY_HEIGHT / stack["xray"].shape[0]
    contours = None
    if show_outline:
        mask = cv2.resize(stack["outline"].astype(np.uint8), None, fx=factor, fy=factor,
                          interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tiles = []
    for caption, image in panels:
        big = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        if contours is not None:
            cv2.drawContours(big, contours, -1, OUTLINE_COLOUR, 1)
        cv2.putText(big, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(big)
    gap = np.full((tiles[0].shape[0], GAP, 3), 255, np.uint8)
    view = np.hstack([tiles[0], gap, tiles[1]])
    hint = "scroll / Up / Down: move through the CT    Space: outline"
    cv2.putText(view, hint, (6, view.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                cv2.LINE_AA)
    return view
