"""VRoom Object-Centric Isolation Pipeline.

This package implements the per-object isolation pipeline that runs after
the ObjectGS scene model has been trained. It is organized in four stages:

    Extract     — pull a single object's pixels and Gaussians out of the
                  trained scene model using ObjectGS masks + Module1 ids.
    Hallucinate — synthesize plausible novel views of the isolated object
                  with an SV3D diffusion prior.
    Train       — fit a fresh per-object Gaussian Splatting model from
                  scratch on the (real + hallucinated) view bundle.
    Re-integrate— stitch the cleaned per-object model back into the scene.

Entrypoints:
    python -m object_isolation.run_pipeline   # full per-object pipeline
    python -m object_isolation.run_training   # training stage only
"""
