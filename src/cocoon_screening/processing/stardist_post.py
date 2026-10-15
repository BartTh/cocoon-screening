"""StarDist 2D pre- and post-processing, without the stardist, csbdeep or scikit-image packages.

The app needs four functions from those packages, but importing any of them pulls in
scipy, scikit-image, numba and llvmlite (~250 MB in the bundle), and stardist's compiled
extension has no wheels beyond Python 3.13. These are faithful ports, written to give
bit-identical label maps; tests/test_stardist_post.py checks that against the originals.
Where scores tie exactly, the order of the survivors (so their label numbers) can differ
between numpy versions, in the original as much as here; the outlines do not.

Ported from (all BSD 3-Clause):
- csbdeep 0.8.2, csbdeep/utils/utils.py: normalize, normalize_mi_ma
- stardist 0.9.2, stardist/nms.py: non_maximum_suppression, _ind_prob_thresh
- stardist 0.9.2, stardist/lib/stardist2d.cpp: c_non_max_suppression_inds
- stardist 0.9.2, stardist/geometry/geom2d.py: dist_to_coord, polygons_to_label
- scikit-image 0.25.2, skimage/draw/_draw.pyx and skimage/_shared/geometry.pyx: polygon
Copyright (c) 2018-2025, Uwe Schmidt, Martin Weigert; (c) 2009-2025 the scikit-image team.
"""

import math
from itertools import accumulate

import numpy as np
import pyclipper


def normalize(x, pmin, pmax, eps=1e-20):
    """Percentile normalisation, as csbdeep.utils.normalize (numpy path, float32)."""
    x = x.astype(np.float32, copy=False)
    mi = np.percentile(x, pmin, keepdims=True).astype(np.float32, copy=False)
    ma = np.percentile(x, pmax, keepdims=True).astype(np.float32, copy=False)
    return (x - mi) / (ma - mi + np.float32(eps))


def non_maximum_suppression(dist, prob, grid, prob_thresh, nms_thresh, b=2):
    """Candidates above prob_thresh, best first; drop any that overlaps a better one.

    Overlap is intersection / smaller area, as in StarDist. Returns points, probs, dists of
    the survivors, in descending probability.
    """
    mask = prob > prob_thresh
    if b > 0:
        inner = np.zeros_like(mask)
        inner[b:-b, b:-b] = True
        mask &= inner
    points = np.stack(np.where(mask), axis=1)
    dist = dist[mask]
    scores = prob[mask]
    # Best first, exactly as StarDist sorts. Among exactly equal scores this order is up to
    # numpy's sort implementation, and it decides which of two tied, overlapping candidates
    # survives (XR_000018 has such a pair). A stable sort would be deterministic but keeps
    # the other one of that pair, so existing label maps would no longer be reproduced;
    # numpy 1.26 and 2.2 agree on which survive on all 51 trays. tests/ catch a change.
    order = np.argsort(scores)[::-1]
    dist, scores, points = dist[order], scores[order], points[order]
    points = points * np.array(grid).reshape((1, 2))
    keep = _nms_keep(np.ascontiguousarray(dist, np.float32),
                     points.astype(np.int32).astype(np.float32), np.float32(nms_thresh))
    return points[keep], scores[keep], dist[keep]


# Below 2**24 every integer is exact in float32, so StarDist's float32 running sum of
# integer cross products equals the exact integer sum; outside that range, emulate it.
_FLOAT32_EXACT = 1 << 24


def _path_area(path):
    """area_from_path: 0.5 * |sum of cross products|, accumulated in float32 like the C++."""
    terms = [x0 * y1 - y0 * x1 for (x0, y0), (x1, y1) in zip(path, path[1:] + path[:1])]
    sums = list(accumulate(terms))
    if max(map(abs, terms)) >= _FLOAT32_EXACT or max(map(abs, sums)) >= _FLOAT32_EXACT:
        total = float(np.cumsum(np.array(terms, np.float32), dtype=np.float32)[-1])
    else:
        total = sums[-1]
    return float(np.float32(0.5 * abs(total)))


def _intersection_area(clipper, path_a, path_b):
    clipper.Clear()
    try:
        clipper.AddPath(path_a, pyclipper.PT_CLIP, True)
        clipper.AddPath(path_b, pyclipper.PT_SUBJECT, True)
    except pyclipper.ClipperException:
        # Clipper's C++ AddPath silently ignores a degenerate path; with one side missing
        # the intersection is empty.
        return 0.0
    result = clipper.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    area = np.float32(0)
    for path in result:
        area = np.float32(area + np.float32(_path_area(path)))
    return float(area)


def _polygon_areas(ix, iy, block=16384):
    """_path_area for every polygon (same float32 sequential sum), in blocks to bound memory."""
    out = np.empty(len(ix), np.float32)
    for start in range(0, len(ix), block):
        x = ix[start:start + block].astype(np.int64)
        y = iy[start:start + block].astype(np.int64)
        terms = x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)
        sums = np.cumsum(terms, axis=1)
        if max(np.abs(terms).max(initial=0), np.abs(sums).max(initial=0)) < _FLOAT32_EXACT:
            total = sums[:, -1].astype(np.float64)
        else:
            total = np.cumsum(terms.astype(np.float32), axis=1, dtype=np.float32)[:, -1].astype(np.float64)
        out[start:start + block] = 0.5 * np.abs(total)
    return out


def _nms_keep(dist, points, threshold):
    """Port of c_non_max_suppression_inds (use_bbox=1, use_kdtree=1)."""
    n_polys, n_rays = dist.shape
    if n_polys == 0:
        return np.zeros(0, bool)
    # Vertices in float32 exactly as the C++ computes them: sinf/cosf of float angles, which
    # equal float64 sin/cos rounded to float32 (numpy's own float32 sin does not).
    angle = np.float32(2 * math.pi / n_rays) * np.arange(n_rays, dtype=np.float32)
    sin = np.sin(angle.astype(np.float64)).astype(np.float32)
    cos = np.cos(angle.astype(np.float64)).astype(np.float32)
    y = points[:, :1] + dist * sin
    x = points[:, 1:] + dist * cos
    # ClipperLib::IntPoint takes cInt: float -> integer truncates toward zero. Pixel
    # coordinates fit int32, which halves the largest arrays (~170k x 32 per tray).
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    # Bounding boxes are float but passed to bbox_intersect as int: truncated.
    bx1, bx2 = x.min(1).astype(np.int64), x.max(1).astype(np.int64)
    by1, by2 = y.min(1).astype(np.int64), y.max(1).astype(np.int64)
    del x, y
    areas = _polygon_areas(ix, iy).astype(np.float64)
    radius = np.maximum(dist.max(1), 0).astype(np.float32)
    max_radius = np.float32(radius.max())
    # For the overlap bounds: extent and perimeter of the integer polygons Clipper sees.
    vx1, vx2 = ix.min(1).astype(np.int64), ix.max(1).astype(np.int64)
    vy1, vy2 = iy.min(1).astype(np.int64), iy.max(1).astype(np.int64)
    perimeter = np.empty(n_polys)
    for start in range(0, n_polys, 16384):
        bx, by = ix[start:start + 16384], iy[start:start + 16384]
        perimeter[start:start + 16384] = np.hypot(np.roll(bx, -1, axis=1) - bx,
                                                  np.roll(by, -1, axis=1) - by).sum(1)

    # Bucket candidates by centre. A polygon whose (truncated) bbox meets i's has its centre
    # within max_radius + 1 of i's bbox; reach is a further pixel beyond that, to be safe.
    reach_box = float(max_radius) + 2
    cell = max(16, int(math.ceil(max_radius)))
    cy = (points[:, 0] // cell).astype(np.int64)
    cx = (points[:, 1] // cell).astype(np.int64)
    buckets = {}
    for idx, key in enumerate(zip(cy.tolist(), cx.tolist())):
        buckets.setdefault(key, []).append(idx)
    buckets = {k: np.array(v) for k, v in buckets.items()}

    # Vertex lists are made on first use and dropped once suppressed: building all of them
    # up front costs several hundred MB of Python ints for ~170k candidates.
    paths = {}

    def path(k):
        if k not in paths:
            paths[k] = list(zip(ix[k].tolist(), iy[k].tolist()))
        return paths[k]

    clipper = pyclipper.Pyclipper()
    suppressed = np.zeros(n_polys, bool)
    threshold = float(threshold)
    for i in range(n_polys - 1):
        if suppressed[i]:
            continue
        r0, r1 = int((by1[i] - reach_box) // cell), int((by2[i] + reach_box) // cell)
        c0, c1 = int((bx1[i] - reach_box) // cell), int((bx2[i] + reach_box) // cell)
        near = [buckets[(r, c)] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if (r, c) in buckets]
        if not near:
            continue
        j = np.concatenate(near)
        # kd-tree radius search (squared L2 in float32, strict) and the bbox test: both only
        # skip pairs that cannot overlap, so they change speed, not the result.
        d2 = (points[j, 0] - points[i, 0]) ** 2 + (points[j, 1] - points[i, 1]) ** 2
        reach = np.float32(max_radius + radius[i])
        j = j[(j > i) & ~suppressed[j] & (d2 < reach * reach)
              & (bx1[j] <= bx2[i]) & (bx1[i] <= bx2[j]) & (by1[j] <= by2[i]) & (by1[i] <= by2[j])]
        if not len(j):
            continue
        smaller = np.minimum(areas[i] + 1e-10, areas[j] + 1e-10)
        # Settle most pairs without clipping (each bound is safe; uncertain pairs are clipped):
        # - Clipper's area is within 0.71 * (sum of perimeters) of the true intersection, as it
        #   rounds crossing points to integers, and the true intersection is at least
        #   A_i + A_j - (bounding box of both). Clearly above the threshold: suppressed.
        # - Every output path lies in the overlap of the two extents, grown by half a pixel
        #   (rectangle R). StarDist sums |area| over paths, holes included, so the total is
        #   at most 2R. Clearly below the threshold: kept.
        union = ((np.maximum(vx2[i], vx2[j]) - np.minimum(vx1[i], vx1[j]))
                 * (np.maximum(vy2[i], vy2[j]) - np.minimum(vy1[i], vy1[j])))
        lower = areas[i] + areas[j] - union - 0.71 * (perimeter[i] + perimeter[j]) - 1
        width = np.minimum(vx2[i], vx2[j]) - np.maximum(vx1[i], vx1[j]) + 1
        height = np.minimum(vy2[i], vy2[j]) - np.maximum(vy1[i], vy1[j]) + 1
        upper = 2 * np.clip(width, 0, None) * np.clip(height, 0, None)
        sure = lower / smaller > threshold * (1 + 1e-6) + 1e-6
        unsure = ~sure & (upper / smaller > threshold * (1 - 1e-6))
        suppressed[j[sure]] = True
        for k in j[unsure].tolist():
            inter = _intersection_area(clipper, path(i), path(k))
            if inter / min(areas[i] + 1e-10, areas[k] + 1e-10) > threshold:
                suppressed[k] = True
        for k in j[sure].tolist():
            paths.pop(k, None)
    return ~suppressed


def dist_to_coord(dist, points):
    """Polygon vertices, shape (n_polys, 2, n_rays): rows then columns."""
    n_rays = dist.shape[1]
    phis = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(np.float32)
    coord += points[..., np.newaxis]
    return coord


def polygons_to_label(dist, points, prob, shape):
    """Rasterise polygons, lowest probability first, so better ones are painted on top.

    Label i + 1 belongs to the i-th polygon as given (descending probability from NMS).
    """
    order = np.argsort(prob, kind="stable")
    coord = dist_to_coord(dist[order], points[order])
    labels = np.zeros(shape, np.int32)
    for i, c in zip(order, coord):
        rr, cc = polygon(c[0], c[1], shape)
        labels[rr, cc] = i + 1
    return labels


def polygon(r, c, shape):
    """Pixels inside or on the boundary of a polygon, as skimage.draw.polygon."""
    minr = int(max(0, r.min()))
    maxr = min(shape[0] - 1, int(math.ceil(r.max())))
    minc = int(max(0, c.min()))
    maxc = min(shape[1] - 1, int(math.ceil(c.max())))
    if maxr < minr or maxc < minc:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    rows, cols = np.mgrid[minr:maxr + 1, minc:maxc + 1]
    inside = _point_in_polygon(c.astype(np.float64), r.astype(np.float64),
                               cols.ravel().astype(np.float64), rows.ravel().astype(np.float64))
    return rows.ravel()[inside], cols.ravel()[inside]


def _point_in_polygon(xp, yp, x, y):
    """Vectorised skimage point_in_polygon: true for inside, vertex or edge.

    Edge k runs from vertex k-1 to vertex k, as in skimage's loop; all edges at once.
    """
    eps = 1e-12
    x0 = xp[:, None] - x[None, :]
    y0 = yp[:, None] - y[None, :]
    x1 = np.roll(x0, 1, axis=0)
    y1 = np.roll(y0, 1, axis=0)
    vertex = ((-eps < x0) & (x0 < eps) & (-eps < y0) & (y0 < eps)).any(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = (x0 * y1 - x1 * y0) / (y1 - y0)
    r_cross = (((y0 > 0) != (y1 > 0)) & (cross > 0)).sum(axis=0)
    l_cross = (((y0 < 0) != (y1 < 0)) & (cross < 0)).sum(axis=0)
    return vertex | ((r_cross & 1) != (l_cross & 1)) | ((r_cross & 1) == 1)
