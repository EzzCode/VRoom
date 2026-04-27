"""
VRoom Target Replenishment — Era3D Novel View Pipeline

Modules:
    objectgs_bridge      — Interface to ObjectGS (model loading, rendering)
    perspective_graph    — Training camera graph and view selection
    coverage_analyzer    — Coverage gap detection for unseen hemispheres
    novel_view_generator — Era3D multi-view generation
    view_alignment       — Coordinate frame alignment (Era3D → Scaffold-GS)
    optimizer            — Fine-tuning with frozen MLPs
"""
