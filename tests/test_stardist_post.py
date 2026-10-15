"""Tests for the StarDist port and tiled inference.

    pip install pytest
    pytest tests

Three layers:
- Port vs original: run only where stardist, csbdeep and scikit-image are installed
  (pip install stardist==0.9.2 csbdeep==0.8.2 on Python <= 3.13). They are not app
  dependencies any more; these tests are how the port was proven bit-identical.
- Tiling: tiled inference against untiled, on synthetic images. Needs only the app.
- Regression: label maps against a frozen reference set, when COCOON_REFERENCE_DIR and
  COCOON_IMAGES_DIR are set (see tests/make_reference.py).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src.cocoon_screening.processing import stardist_post as sp
from src.cocoon_screening.processing.stardist import (
    DIVISOR, GRID, NMS_THRESH, PROB_THRESH, StardistPredictor, _tile_spans)


def synthetic_tray(shape, seed=0, n=60):
    """Bright blobs on a dark, noisy background: enough to give real NMS candidates."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    image = rng.normal(40, 8, shape)
    for _ in range(n):
        r, c = rng.uniform(0, shape[0]), rng.uniform(0, shape[1])
        a, b = rng.uniform(12, 30, 2)
        image += 150 * np.exp(-(((rows - r) / a) ** 2 + ((cols - c) / b) ** 2) ** 2)
    return np.clip(image, 0, 255).astype(np.uint8)


def assert_same_detections(expected, actual):
    """points, prob, dist equal, except that exactly tied scores may come in another order
    (that order is up to numpy's sort implementation, in StarDist as here)."""
    (pe, se, de), (pa, sa, da) = expected, actual
    assert len(pe) == len(pa)
    assert np.array_equal(se, sa)                       # sorted descending in both
    key = lambda p, s: np.lexsort((p[:, 1], p[:, 0], -s))
    ie, ia = key(pe, se), key(pa, sa)
    assert np.array_equal(pe[ie], pa[ia]) and np.array_equal(de[ie], da[ia])
    moved = np.flatnonzero(np.any(pe != pa, axis=1))
    for m in moved:                                     # only within a run of equal scores
        assert np.sum(se == se[m]) > 1


def assert_same_segmentation(expected, actual):
    """Identical outlines; instance numbers may differ (a one-to-one renumbering)."""
    assert expected.shape == actual.shape
    pairs = np.unique(np.stack([expected.ravel().astype(np.int64), actual.ravel().astype(np.int64)]), axis=1)
    assert np.all((pairs[0] == 0) == (pairs[1] == 0)), "foreground differs"
    fg = pairs[:, pairs[0] > 0]
    assert len(np.unique(fg[0])) == len(np.unique(fg[1])) == fg.shape[1], "not a one-to-one renumbering"


@pytest.fixture(scope="module")
def untiled():
    return StardistPredictor(tile_size=None)


# --- tiling -----------------------------------------------------------------------------

@pytest.mark.parametrize("length", [16, 160, 1000, 2656, 2160, 3001 // 16 * 16])
@pytest.mark.parametrize("tile", [128, 1024])
def test_tile_spans_cover_exactly_on_divisor_bounds(length, tile):
    spans = _tile_spans(length, tile)
    assert spans[0][0] == 0 and spans[-1][1] == length
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(start % DIVISOR == 0 for start, _ in spans)
    assert all(end - start <= max(tile, DIVISOR) for start, end in spans)


@pytest.mark.parametrize("shape,tile", [((700, 900), 256), ((531, 1003), 160), ((300, 300), 1024)])
def test_tiled_network_matches_untiled(untiled, shape, tile):
    image = sp.normalize(synthetic_tray(shape).astype(np.float32), 1, 99.8)
    prob_u, dist_u = untiled.network(image)
    prob_t, dist_t = StardistPredictor(tile_size=tile).network(image)
    assert prob_t.shape == prob_u.shape and dist_t.shape == dist_u.shape
    # Every output pixel sees exactly its untiled input, so the result is bit-identical
    # (measured on real trays too, with 3x3 and 6x5 tiles).
    assert np.array_equal(prob_t, prob_u)
    assert np.array_equal(dist_t, dist_u)


@pytest.mark.parametrize("shape,tile", [((700, 900), 256), ((531, 1003), 160)])
def test_tiled_labels_match_untiled(untiled, shape, tile):
    image = sp.normalize(synthetic_tray(shape, seed=1).astype(np.float32), 1, 99.8)
    labels_u, _ = untiled._segment(image)
    labels_t, _ = StardistPredictor(tile_size=tile)._segment(image)
    assert labels_u.max() > 10
    assert np.array_equal(labels_t, labels_u)


def test_too_small_margin_would_be_caught(untiled, monkeypatch):
    """Guard on the test itself: with no context the seams must show."""
    import src.cocoon_screening.processing.stardist as module
    image = sp.normalize(synthetic_tray((512, 512)).astype(np.float32), 1, 99.8)
    prob_u, _ = untiled.network(image)
    monkeypatch.setattr(module, "TILE_MARGIN", 0)
    prob_t, _ = StardistPredictor(tile_size=256).network(image)
    assert np.abs(prob_t - prob_u).max() > 1e-3


# --- port vs original -------------------------------------------------------------------

def _require_originals():
    pytest.importorskip("stardist")
    pytest.importorskip("csbdeep")
    pytest.importorskip("skimage")


def test_normalize_matches_csbdeep():
    _require_originals()
    from csbdeep.utils import normalize
    for seed in range(3):
        image = synthetic_tray((400, 333), seed).astype(np.float32)
        assert np.array_equal(sp.normalize(image, 1, 99.8), normalize(image, 1, 99.8))


@pytest.mark.parametrize("seed", range(4))
def test_nms_and_labels_match_stardist(untiled, seed):
    _require_originals()
    from stardist.geometry import polygons_to_label
    from stardist.nms import non_maximum_suppression
    image = sp.normalize(synthetic_tray((640, 704), seed, n=120).astype(np.float32), 1, 99.8)
    prob, dist = untiled.network(image)
    expected = non_maximum_suppression(dist, prob, grid=GRID, prob_thresh=PROB_THRESH,
                                       nms_thresh=NMS_THRESH)
    actual = sp.non_maximum_suppression(dist, prob, GRID, PROB_THRESH, NMS_THRESH)
    assert len(expected[0]) > 10
    assert_same_detections(expected, actual)
    points, probi, disti = expected
    assert np.array_equal(sp.polygons_to_label(disti, points, probi, image.shape),
                          polygons_to_label(disti, points, prob=probi, shape=image.shape))


def test_polygon_matches_skimage_on_awkward_shapes():
    _require_originals()
    from skimage.draw import polygon
    rng = np.random.default_rng(7)
    shape = (60, 50)
    cases = [
        (np.array([10, 10, 30, 30.]), np.array([10, 30, 30, 10.])),        # vertices on pixels
        (np.array([-5, -5, 20, 20.]), np.array([-5, 70, 70, -5.])),        # beyond the image
        (np.array([5.5, 5.5, 5.5]), np.array([1, 2, 3.])),                 # degenerate line
    ]
    for _ in range(200):
        n = rng.integers(3, 40)
        phi = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(2, 25, n)
        cases.append((rng.uniform(0, 60) + rad * np.sin(phi), rng.uniform(0, 50) + rad * np.cos(phi)))
    for r, c in cases:
        r, c = r.astype(np.float32), c.astype(np.float32)
        expected = np.zeros(shape, bool); expected[polygon(r, c, shape)] = True
        actual = np.zeros(shape, bool); actual[sp.polygon(r, c, shape)] = True
        assert np.array_equal(actual, expected)


# --- regression against a frozen reference ----------------------------------------------

REFERENCE = os.environ.get("COCOON_REFERENCE_DIR")
IMAGES = os.environ.get("COCOON_IMAGES_DIR")


@pytest.mark.skipif(not (REFERENCE and IMAGES), reason="set COCOON_REFERENCE_DIR and COCOON_IMAGES_DIR")
def test_label_maps_match_reference():
    predictor = StardistPredictor()
    references = sorted(Path(REFERENCE).glob("XR_*.npz"))
    references = [r for r in references if not r.stem.endswith("_raw")]
    assert references
    for ref in references:
        expected = np.load(ref)
        labels, details = predictor.predict(Path(IMAGES) / f"{ref.stem}.jpg")
        assert_same_segmentation(expected["labels"], labels)
        assert {tuple(p) for p in details["points"]} == {tuple(p) for p in expected["points"]}, ref.stem
        assert np.array_equal(np.sort(details["prob"]), np.sort(expected["prob"])), ref.stem
