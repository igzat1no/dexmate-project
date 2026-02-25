"""
Depth image → per-object 3D point cloud extraction with table removal and denoising.

Pipeline:
  1. Back-project full depth image → global scene point cloud.
  2. RANSAC plane fitting to detect the dominant table surface.
  3. Build a table-surface coordinate frame (u, v, normal).
  4. Remove table-plane inliers so object centroids are not biased.
  5. For each object mask, extract its 3D points (table-excluded),
     apply statistical outlier removal (SOR) to discard isolated noise,
     compute yaw via PCA on the table-plane projection,
     and compute an oriented bounding box (OBB).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
from loguru import logger


# ── camera intrinsics ──────────────────────────────────────────────────
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    def pixel_to_3d(self, u: np.ndarray, v: np.ndarray, z: np.ndarray):
        """Back-project pixel coords (u, v) with depth z → (X, Y, Z) in camera frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)


REALSENSE_R200 = CameraIntrinsics(
    fx=618.1409912109375,
    fy=618.1409912109375,
    cx=311.46591186523438,
    cy=236.29269409179688,
    width=640,
    height=480,
)

DEFAULT_DEPTH_SCALE = 1e-4  # meters per raw depth unit (YCB-M: 0.1 mm)


# ── global scene helpers ───────────────────────────────────────────────
def backproject_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project the full depth image to a 3D point cloud.

    Returns:
        points_3d: (H, W, 3) float64 array; pixels with depth==0 have z=0.
        valid_mask: (H, W) bool – True where depth > 0.
    """
    H, W = depth.shape[:2]
    v_map, u_map = np.mgrid[0:H, 0:W].astype(np.float64)
    depth_m = depth.astype(np.float64) * depth_scale
    valid_mask = depth > 0

    x = (u_map - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v_map - intrinsics.cy) * depth_m / intrinsics.fy
    points_3d = np.stack([x, y, depth_m], axis=-1)
    return points_3d, valid_mask


def fit_table_plane(
    points_3d: np.ndarray,
    valid_mask: np.ndarray,
    ransac_distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    """
    Fit a dominant plane (table) in the scene via RANSAC.

    Returns:
        plane_model: (4,) array [a, b, c, d] s.t. ax+by+cz+d=0, or None.
        inlier_ratio: fraction of valid points that are on the plane.
        table_pixel_mask: (H, W) bool – True for pixels belonging to the table.
    """
    H, W = valid_mask.shape
    flat_pts = points_3d[valid_mask]  # (M, 3)
    if len(flat_pts) < 100:
        logger.warning("Too few valid depth pixels ({}) for RANSAC", len(flat_pts))
        return None, 0.0, np.zeros((H, W), dtype=bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flat_pts)

    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=ransac_distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane_model = np.asarray(plane_model)
    inlier_ratio = len(inlier_indices) / len(flat_pts)

    valid_flat_idx = np.flatnonzero(valid_mask.ravel())
    table_flat = valid_flat_idx[inlier_indices]
    table_pixel_mask = np.zeros(H * W, dtype=bool)
    table_pixel_mask[table_flat] = True
    table_pixel_mask = table_pixel_mask.reshape(H, W)

    logger.info(
        "RANSAC table plane: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]  "
        "inlier ratio: {:.1%} ({} pts)",
        *plane_model,
        inlier_ratio,
        len(inlier_indices),
    )
    return plane_model, inlier_ratio, table_pixel_mask


# ── table coordinate frame ─────────────────────────────────────────────
def build_table_frame(
    plane_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an orthonormal coordinate frame on the table surface.

    The normal is oriented so it points towards the camera (origin in
    camera frame), serving as the "up" direction from the table.

    Returns:
        table_u:  (3,) unit vector — "right" on the table (projection of
                  camera X-axis onto the plane).
        table_v:  (3,) unit vector — "forward" on the table (normal × u).
        table_up: (3,) unit vector — surface normal pointing towards camera.
    """
    n = np.array(plane_model[:3], dtype=np.float64)
    n = n / np.linalg.norm(n)
    d = float(plane_model[3])

    # Ensure the normal points towards the camera (origin).
    # Signed distance of origin from plane = d / ||n|| (n already unit).
    # If d < 0, origin is on the negative side → flip normal.
    if d < 0:
        n = -n

    # Project camera X-axis onto the table plane to define "right".
    cam_x = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, cam_x)) > 0.95:
        cam_x = np.array([0.0, 1.0, 0.0])
    u = cam_x - np.dot(cam_x, n) * n
    u = u / np.linalg.norm(u)

    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    logger.info(
        "Table frame — u(right)=[{:.3f},{:.3f},{:.3f}]  "
        "v(fwd)=[{:.3f},{:.3f},{:.3f}]  up=[{:.3f},{:.3f},{:.3f}]",
        *u, *v, *n,
    )
    return u, v, n


# ── PCA yaw + oriented bounding box ───────────────────────────────────
def _compute_yaw_and_obb(
    pts: np.ndarray,
    table_u: np.ndarray,
    table_v: np.ndarray,
    table_up: np.ndarray,
) -> tuple[float, list[float]]:
    """
    Compute the yaw (rotation on the table plane) and oriented bounding
    box size for a single object.

    Steps:
      1. Project points onto the table-plane 2D frame (u, v).
      2. PCA on the 2D projection → first eigenvector = main axis.
      3. yaw = angle from u-axis to the main axis.
      4. Rotate the 2D points by –yaw to align with the main axis,
         then compute axis-aligned extent → (length, width).
      5. height = extent along the table normal.

    Returns:
        yaw:      Angle in radians (–π, π].
        obb_size: [length, width, height] in metres.
    """
    if len(pts) < 3:
        return 0.0, [0.0, 0.0, 0.0]

    centroid = pts.mean(axis=0)
    pts_c = pts - centroid

    proj_u = pts_c @ table_u
    proj_v = pts_c @ table_v
    proj_h = pts_c @ table_up

    pts_2d = np.column_stack([proj_u, proj_v])

    cov = np.cov(pts_2d, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    main_dir = eigenvectors[:, -1]  # largest eigenvalue = last column

    yaw = float(np.arctan2(main_dir[1], main_dir[0]))

    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    pts_aligned = pts_2d @ rot.T

    length = float(pts_aligned[:, 0].max() - pts_aligned[:, 0].min())
    width = float(pts_aligned[:, 1].max() - pts_aligned[:, 1].min())
    height = float(proj_h.max() - proj_h.min())

    return yaw, [round(length, 5), round(width, 5), round(height, 5)]


# ── per-object denoising ──────────────────────────────────────────────
def _statistical_outlier_removal(
    pts: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Remove isolated noise points via Statistical Outlier Removal (SOR).

    Preserves all spatially coherent clusters (important when objects
    are partially occluded and split into multiple disconnected groups).
    """
    if len(pts) < nb_neighbors:
        return pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio,
    )
    if len(inlier_idx) == 0:
        return pts
    return pts[inlier_idx]


# ── main extraction pipeline ──────────────────────────────────────────
def extract_object_point_clouds(
    depth: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    intrinsics: CameraIntrinsics = REALSENSE_R200,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    min_points: int = 10,
    ransac_dist: float = 0.01,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    include_table: bool = True,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """
    Full 3D lifting pipeline: back-project → table removal → per-object
    SOR denoising → PCA yaw → oriented bounding box.

    Returns:
        results:     List of dicts.  Each dict contains:
                       object_id, label, confidence, centroid_xyz,
                       size_xyz ([length, width, height] in OBB frame),
                       yaw, bbox, num_points.
                     The table object additionally stores
                       table_u_axis, table_v_axis, table_normal
                     so the resolver can reconstruct per-object axes.
        raw_clouds:  Dict mapping label → (N, 3) cleaned point cloud.
    """
    # ── 1. Global back-projection ──
    points_3d, valid_mask = backproject_depth(depth, intrinsics, depth_scale)

    # ── 2. RANSAC table fitting ──
    plane_model, inlier_ratio, table_mask = fit_table_plane(
        points_3d, valid_mask, ransac_distance_threshold=ransac_dist,
    )
    non_table_mask = valid_mask & (~table_mask)

    # ── 3. Table coordinate frame ──
    if plane_model is not None:
        table_u, table_v, table_up = build_table_frame(plane_model)
    else:
        table_u = np.array([1.0, 0.0, 0.0])
        table_v = np.array([0.0, 1.0, 0.0])
        table_up = np.array([0.0, 0.0, -1.0])

    results = []
    raw_clouds: dict[str, np.ndarray] = {}
    next_id = 0

    # ── Table as an object ──
    if include_table and table_mask.any():
        table_pts = points_3d[table_mask]
        centroid_t = table_pts.mean(axis=0).tolist()
        size_t = (table_pts.max(axis=0) - table_pts.min(axis=0)).tolist()
        results.append({
            "object_id": next_id,
            "label": "table",
            "confidence": round(float(inlier_ratio), 4),
            "centroid_xyz": [round(c, 5) for c in centroid_t],
            "size_xyz": [round(s, 5) for s in size_t],
            "yaw": 0.0,
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "num_points": len(table_pts),
            "table_u_axis": [round(float(v), 6) for v in table_u],
            "table_v_axis": [round(float(v), 6) for v in table_v],
            "table_normal": [round(float(v), 6) for v in table_up],
        })
        raw_clouds["table"] = table_pts
        next_id += 1

    # ── 4. Per-object extraction ──
    for i in range(len(labels)):
        obj_mask = masks[i].astype(bool) if masks[i].dtype != bool else masks[i]

        obj_valid = obj_mask & non_table_mask
        n_raw = int((obj_mask & valid_mask).sum())
        n_no_table = int(obj_valid.sum())

        if n_no_table < min_points:
            obj_valid = obj_mask & valid_mask
            n_no_table = int(obj_valid.sum())
            if n_no_table < min_points:
                logger.debug(
                    "Skipping '{}': only {} pts after table removal (raw={})",
                    labels[i], n_no_table, n_raw,
                )
                continue

        pts = points_3d[obj_valid]

        # ── 5. Statistical outlier removal ──
        pts_clean = _statistical_outlier_removal(
            pts, nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio,
        )

        centroid = pts_clean.mean(axis=0).tolist()

        # ── 6. PCA yaw + oriented bounding box ──
        yaw, obb_size = _compute_yaw_and_obb(
            pts_clean, table_u, table_v, table_up,
        )

        box = boxes[i].tolist() if hasattr(boxes[i], "tolist") else list(boxes[i])
        lbl = str(labels[i])

        logger.debug(
            "'{}': raw={} → no_table={} → clean={} pts  "
            "obb=[{:.3f}, {:.3f}, {:.3f}] m  yaw={:.2f}°",
            lbl, n_raw, n_no_table, len(pts_clean),
            *obb_size, np.degrees(yaw),
        )

        cloud_key = lbl
        suffix = 2
        while cloud_key in raw_clouds:
            cloud_key = f"{lbl}_{suffix}"
            suffix += 1
        raw_clouds[cloud_key] = pts_clean

        results.append({
            "object_id": next_id,
            "label": lbl,
            "confidence": round(float(confidences[i]), 4),
            "centroid_xyz": [round(c, 5) for c in centroid],
            "size_xyz": obb_size,
            "yaw": round(float(yaw), 5),
            "bbox": [round(b, 1) for b in box],
            "num_points": len(pts_clean),
        })
        next_id += 1

    return results, raw_clouds


# ── PLY export ─────────────────────────────────────────────────────────
_OBJECT_PALETTE = np.array([
    [230,  25,  75], [ 60, 180,  75], [  0, 130, 200], [255, 225,  25],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128],
], dtype=np.uint8)

TABLE_COLOR = np.array([200, 200, 200], dtype=np.uint8)


def export_scene_ply(
    raw_clouds: dict[str, np.ndarray],
    save_path: str,
    downsample_table: int = 4,
) -> None:
    """Merge all object point clouds into a single coloured PLY file."""
    all_pts = []
    all_colors = []
    color_idx = 0

    for label, pts in raw_clouds.items():
        if len(pts) == 0:
            continue
        if label == "table":
            pts_ds = pts[::downsample_table]
            all_pts.append(pts_ds)
            all_colors.append(np.tile(TABLE_COLOR, (len(pts_ds), 1)))
        else:
            color = _OBJECT_PALETTE[color_idx % len(_OBJECT_PALETTE)]
            color_idx += 1
            all_pts.append(pts)
            all_colors.append(np.tile(color, (len(pts), 1)))

    if not all_pts:
        logger.warning("No points to export.")
        return

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors.astype(np.float64) / 255.0)

    o3d.io.write_point_cloud(save_path, pcd)
    logger.info(
        "Exported coloured PLY → {}  ({} objects, {} total pts)",
        save_path, len(raw_clouds), len(merged_pts),
    )
