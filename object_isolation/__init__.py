"""Object-Centric Isolation Pipeline (Zero123++ Healing).

Extract a single object from a trained ObjectGS room model, hallucinate the
unseen hemisphere with Zero123++, calibrate the metric scale of the novel
views against COLMAP sparse points, train a standalone 2DGS from scratch,
and reintegrate the result back into the room.

Sibling package to ``target_replenishment/`` — does not import its
optimizer/finetune modules. Reuses ``target_replenishment.core.objectgs_bridge``
for ObjectGS model loading and rendering only.
"""

__all__ = []
