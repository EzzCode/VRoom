"""VRoom Target Replenishment core — Deterministic Shrinkwrap pipeline.

Modules:
    objectgs_bridge        — Interface to ObjectGS (model loading, rendering)
    surface_extraction     — Stage A: topological dense-surface filter
    directional_shrinkwrap — Stage B: 6-sided projected shrinkwrap walls
    knn_initializer        — Stage C: normal-aware seed init
    render_floater_pruning — Stage A/E: render-space disconnected blob prune
    diagnostics            — Before/after renders + AABB overlays
"""
