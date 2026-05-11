"""Pluggable Diffusion-Prior Backends for Novel-View Hallucination.

This subpackage exposes a small abstract interface
(:class:`object_isolation.core.diffusion_priors.base.DiffusionPriorBackend`)
plus one concrete implementation (Stable Video 3D, see :mod:`sv3d`). Adding a
new prior is a matter of subclassing the base class and registering it where
the pipeline picks the backend.
"""
