import logging
import cv2
import numpy as np
import re

from ModuleTBD.utils.gstrain_wrapper import make_camera, render_rgba, render_rgba

logger = logging.getLogger(__name__)

def normalize(vector: np.ndarray):
    eps = 1e-8
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"Cannot normalize zero or non-finite vector: {vector}")
    return (vector / norm).astype(np.float32)

def find_image(images_dir, img_name):
    if not images_dir.exists():
        return None
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        c = images_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    for f in images_dir.iterdir():
        if f.is_file() and f.stem == img_name:
            return f
    return None


def find_id_map(id_map_dir, img_name):
    if not id_map_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        c = id_map_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    # Replica-style rename
    renamed = img_name.replace("_rgb_", "_semantic_instance_").replace("rgb", "semantic_instance")
    if renamed != img_name:
        for ext in (".png", ".jpg", ".jpeg"):
            c = id_map_dir / f"{renamed}{ext}"
            if c.exists():
                return c
    # Trailing-digit fuzzy match
    m = re.search(r"(\d+)$", img_name)
    if m:
        suffix = m.group(1)
        for f in id_map_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                m2 = re.search(r"(\d+)$", f.stem)
                if m2 and m2.group(1) == suffix:
                    return f
    return None

def auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir, n_probe=5, tau_alpha=0.4):
    """Vote across cameras to find the Module1 instance id that best matches the ObjectGS silhouette."""
    if not id_map_dir.exists():
        return None
    indices = list(scope.visible_cam_indices)
    if len(indices) > n_probe:
        indices = indices[::max(1, len(indices) // n_probe)][:n_probe]

    votes = {}
    for ci in indices:
        cam_p = scope.cameras[ci]
        id_map_path = find_id_map(id_map_dir, cam_p["image_name"])
        if id_map_path is None:
            continue
        id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            continue
        cam = make_camera(cam_p["R"], cam_p["T"], cam_p["K"], cam_p["width"], cam_p["height"])
        alpha = render_rgba(gaussians, cam, pipe_config,
                            object_label_id=scope.object_label_id)["alpha"].detach().cpu().numpy()
        m_objgs = (alpha[0] if alpha.ndim == 3 else alpha) > tau_alpha
        H, W = m_objgs.shape
        if id_map.shape[:2] != (H, W):
            id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)
        for uid in np.unique(id_map):
            if int(uid) == 0:
                continue
            m_real = (id_map == uid)
            inter = float(np.logical_and(m_real, m_objgs).sum())
            union = float(np.logical_or(m_real, m_objgs).sum())
            votes[int(uid)] = votes.get(int(uid), 0.0) + inter / max(union, 1.0)

    if not votes:
        return None
    winner, best_score = max(votes.items(), key=lambda kv: kv[1])
    logger.info("Auto-resolved module1_id=%d (top votes: %s)", winner,
                {k: round(v, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:5]})
    return winner if best_score > 0 else None
