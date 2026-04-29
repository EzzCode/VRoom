"""
Render-space floater pruning.

This pass is intentionally narrow: render the isolated object from several
comparison cameras, find alpha-mask components disconnected from the main body,
map those pixels back to parent anchors, then vote-prune anchors that repeatedly
appear only in disconnected blobs.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from target_replenishment.core.objectgs_bridge import (
    build_anchor_id_map,
    render_view,
)


@dataclass
class RenderFloaterPruneResult:
    n_views: int
    n_candidate_anchors: int
    n_pruned: int
    prune_indices: np.ndarray
    vote_threshold: int

    def to_dict(self) -> dict:
        return {
            "n_views": int(self.n_views),
            "n_candidate_anchors": int(self.n_candidate_anchors),
            "n_pruned": int(self.n_pruned),
            "vote_threshold": int(self.vote_threshold),
        }


def find_render_space_floaters(
    gaussians,
    pipe_config,
    object_id: int,
    cameras: list,
    alpha_threshold: float = 0.03,
    close_kernel: int = 9,
    min_blob_area_px: int = 24,
    vote_threshold: int = 2,
) -> RenderFloaterPruneResult:
    """Detect target anchors that render as disconnected 2D blobs.

    The main object body is the largest connected alpha component in each view.
    Anchors whose pixels appear in non-main components get votes. An anchor is
    considered a floater only if it is voted in at least ``vote_threshold``
    views. This makes the pass resistant to one bad projection.
    """
    if not cameras:
        return RenderFloaterPruneResult(0, 0, 0, np.zeros(0, dtype=np.int64), vote_threshold)

    labels = gaussians.label_ids.detach().squeeze(-1).cpu().numpy()
    target_mask = labels == int(object_id)
    vote_counts: dict[int, int] = {}
    bg = torch.ones(3, dtype=torch.float32, device="cuda")
    kernel_size = max(1, int(close_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for cam in cameras:
        render_result = render_view(
            gaussians,
            cam,
            pipe_config,
            bg_color=bg,
            object_label_id=object_id,
        )
        alpha = render_result["alpha"]
        if alpha is None:
            continue
        alpha_np = alpha.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
        mask = (alpha_np > alpha_threshold).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 2:
            continue

        areas = stats[1:, cv2.CC_STAT_AREA]
        main_comp = 1 + int(np.argmax(areas))
        floater_comp_ids = [
            comp_id for comp_id in range(1, n_labels)
            if comp_id != main_comp and stats[comp_id, cv2.CC_STAT_AREA] >= min_blob_area_px
        ]
        if not floater_comp_ids:
            continue

        anchor_map = build_anchor_id_map(
            render_result,
            H=int(cam.image_height),
            W=int(cam.image_width),
            n_anchors=int(gaussians._anchor.shape[0]),
        )
        floater_pixels = np.isin(comp_labels, np.asarray(floater_comp_ids, dtype=np.int32))
        anchor_ids = np.unique(anchor_map[floater_pixels])
        for anchor_id in anchor_ids:
            anchor_id = int(anchor_id)
            if anchor_id < 0 or anchor_id >= labels.shape[0]:
                continue
            if target_mask[anchor_id]:
                vote_counts[anchor_id] = vote_counts.get(anchor_id, 0) + 1

    candidates = np.array(sorted(vote_counts.keys()), dtype=np.int64)
    if candidates.size == 0:
        pruned = candidates
    else:
        pruned = np.array(
            [idx for idx in candidates if vote_counts[int(idx)] >= vote_threshold],
            dtype=np.int64,
        )
    return RenderFloaterPruneResult(
        n_views=len(cameras),
        n_candidate_anchors=int(candidates.size),
        n_pruned=int(pruned.size),
        prune_indices=pruned,
        vote_threshold=vote_threshold,
    )
