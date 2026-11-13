"""Smooth shading normals for cooked terrain cells.

Terrain meshes carry positions only (no normal stream); in game the cell is lit
from its baked ``-normal.xbm``. The cooked triangulation is a decimated
heightfield that contains two kinds of artifacts which are invisible in game but
turn black/white once Blender derives normals from the faces:

* crack fillers: triangles whose three vertices are collinear in XY. They are
  vertical slivers that close T-junctions between areas of different density
  (mostly along the cell border), and their face normal is horizontal.
* fold-overs: a few small triangles wound opposite to the rest of the cell.

Both are kept (removing fillers would open cracks), but they are excluded when
the vertex normals are accumulated. Vertices shared by neighbouring chunks are
welded for the accumulation so the shading stays continuous across chunk seams.
"""

import numpy as np

from ...blender.custom_normals import set_custom_normals_from_vertices
from ...blender.mesh import is_mesh
from ...core.log import get_logger

_LOG = get_logger(__name__)

# Doubled XY-projected area (m^2) below which a triangle is treated as a vertical
# crack filler. Real terrain triangles sit on a 0.5 m grid (doubled area >= 0.25);
# fillers come out around 1e-6 because of position quantization.
_FILLER_DOUBLE_AREA_XY = 1.0e-3
# Positions closer than this (m) are welded when accumulating normals across chunks.
_WELD_DISTANCE = 1.0e-3


def _triangles(mesh):
    vertex_count = len(mesh.vertices)
    positions = np.empty(vertex_count * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", positions)
    mesh.calc_loop_triangles()
    triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", triangles)
    return positions.reshape(vertex_count, 3), triangles.reshape(-1, 3)


def _matrix_key(obj):
    matrix = getattr(obj, "matrix_world", None)
    if matrix is None:
        return None
    return tuple(round(float(value), 6) for row in matrix for value in row)


def _terrain_normals(meshes):
    """Return per-mesh vertex normals for meshes that share one object space."""
    arrays = [_triangles(mesh) for mesh in meshes]
    offsets = np.cumsum([0] + [len(positions) for positions, _ in arrays])
    positions = np.concatenate([p for p, _ in arrays]) if arrays else np.empty((0, 3))
    if not len(positions):
        return [None] * len(meshes), {}
    triangles = np.concatenate(
        [tris + offsets[index] for index, (_, tris) in enumerate(arrays)]
    )

    # Weld coincident chunk-border vertices so seams accumulate the same normal.
    keys = np.round(positions / _WELD_DISTANCE).astype(np.int64)
    _, welded = np.unique(keys, axis=0, return_inverse=True)
    welded = welded.reshape(-1)
    welded_count = int(welded.max()) + 1

    corners = positions[triangles]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area_xy = face[:, 2]
    # The cooked winding is consistent per cell; take the majority as "up".
    orientation = 1.0 if double_area_xy.sum() >= 0.0 else -1.0
    oriented = double_area_xy * orientation
    filler = np.abs(double_area_xy) < _FILLER_DOUBLE_AREA_XY
    folded = (~filler) & (oriented < 0.0)
    usable = ~(filler | folded)

    accumulated = np.zeros((welded_count, 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(accumulated, welded[triangles[usable, corner]], face[usable])

    lengths = np.linalg.norm(accumulated, axis=1)
    missing = lengths <= 1.0e-12
    if np.any(missing):
        # Vertices touched only by fillers/fold-overs (typically the far end of a
        # border filler) borrow the normals of the vertices they share faces with.
        unit = np.zeros_like(accumulated)
        unit[~missing] = accumulated[~missing] / lengths[~missing, None]
        welded_triangles = welded[triangles]
        borrowed = np.zeros_like(accumulated)
        for corner in range(3):
            for other in range(3):
                if other != corner:
                    np.add.at(
                        borrowed,
                        welded_triangles[:, corner],
                        unit[welded_triangles[:, other]],
                    )
        accumulated[missing] = borrowed[missing]
        lengths = np.linalg.norm(accumulated, axis=1)
        still_missing = lengths <= 1.0e-12
        accumulated[still_missing] = (0.0, 0.0, orientation)
        lengths[still_missing] = 1.0
    normals = (accumulated / lengths[:, None])[welded]

    stats = {
        "fillers": int(filler.sum()),
        "folded": int(folded.sum()),
        "orphans": int(missing.sum()),
    }
    return [
        normals[offsets[index] : offsets[index + 1]] for index in range(len(meshes))
    ], stats


def apply_terrain_normals(direct_result):
    """Give terrain cell objects smooth custom normals that ignore cooked artifacts."""
    groups = {}
    for obj in direct_result.get("objects") or ():
        mesh = getattr(obj, "data", None)
        if not is_mesh(obj) or mesh is None or not len(mesh.vertices):
            continue
        if any(mesh is existing for existing, _ in groups.get(_matrix_key(obj), ())):
            continue
        groups.setdefault(_matrix_key(obj), []).append((mesh, obj))

    updated = 0
    totals = {"fillers": 0, "folded": 0, "orphans": 0}
    for members in groups.values():
        meshes = [mesh for mesh, _ in members]
        normals_per_mesh, stats = _terrain_normals(meshes)
        for key, value in stats.items():
            totals[key] += value
        for mesh, normals in zip(meshes, normals_per_mesh):
            if normals is None:
                continue
            polygon_count = len(mesh.polygons)
            if polygon_count:
                mesh.polygons.foreach_set(
                    "use_smooth", np.ones(polygon_count, dtype=np.bool_)
                )
            if set_custom_normals_from_vertices(
                mesh, normals, label=f"terrain mesh {mesh.name!r}"
            ):
                mesh.update()
                updated += 1

    if updated:
        _LOG.debug(
            "Terrain normals for %d submesh(es): ignored %d crack filler(s), "
            "%d fold-over(s); %d vertex(es) borrowed neighbour normals",
            updated,
            totals["fillers"],
            totals["folded"],
            totals["orphans"],
        )
    return {"updated": updated, **totals}
