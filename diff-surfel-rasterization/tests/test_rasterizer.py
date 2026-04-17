"""
Test suite for the modified diff-surfel-rasterization (Object-GS rasterizer).

Run with:
    python tests/test_rasterizer.py
or:
    python -m pytest tests/test_rasterizer.py -v
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_gaussians(N=64, device="cuda"):
    """Create a minimal set of random Gaussians for testing."""
    torch.manual_seed(42)
    means3D   = torch.randn(N, 3, device=device) * 0.5          # near origin
    means3D[:, 2] += 3.0                                         # push in front of camera
    scales    = torch.rand(N, 2, device=device) * 0.1 + 0.01    # small positive scales
    rots      = torch.randn(N, 4, device=device)
    rots      = F.normalize(rots, dim=-1)
    opacities = torch.sigmoid(torch.randn(N, 1, device=device))
    colors    = torch.rand(N, 3, device=device)                  # precomputed RGB
    return means3D, scales, rots, opacities, colors

def make_camera(H=64, W=64, device="cuda"):
    """Simple camera looking down -Z."""
    viewmatrix = torch.eye(4, device=device)   # identity world-to-cam
    # projmatrix: standard OpenGL-style
    znear, zfar = 0.01, 100.0
    fovx = fovy = math.pi / 3                  # 60 degree FoV
    tanfovx = math.tan(fovx / 2)
    tanfovy = math.tan(fovy / 2)

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0

    full_proj = viewmatrix.T @ P.T
    campos    = torch.zeros(3, device=device)
    return viewmatrix, full_proj, campos, tanfovx, tanfovy, H, W

def make_rasterizer(H, W, tanfovx, tanfovy, viewmatrix, full_proj, campos,
                    bg, sh_degree=0, device="cuda"):
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    return GaussianRasterizer(settings)

def PASS(name): print(f"  ✓  {name}")
def FAIL(name, msg): print(f"  ✗  {name}: {msg}"); raise AssertionError(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Import
# ─────────────────────────────────────────────────────────────────────────────

def test_import():
    """The package imports correctly and exposes the expected symbols."""
    from diff_surfel_rasterization import (
        GaussianRasterizer,
        GaussianRasterizationSettings,
        rasterization_2dgs,
        _next_supported,
        _SUPPORTED_CHANNELS,
    )
    assert callable(rasterization_2dgs)
    assert _SUPPORTED_CHANNELS == [1, 3, 4, 8, 16, 32]
    PASS("import")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: _next_supported helper
# ─────────────────────────────────────────────────────────────────────────────

def test_next_supported():
    from diff_surfel_rasterization import _next_supported
    assert _next_supported(1)  == 1
    assert _next_supported(2)  == 3
    assert _next_supported(3)  == 3
    assert _next_supported(4)  == 4
    assert _next_supported(5)  == 8
    assert _next_supported(7)  == 8
    assert _next_supported(8)  == 8
    assert _next_supported(9)  == 16
    assert _next_supported(16) == 16
    assert _next_supported(17) == 32
    assert _next_supported(32) == 32
    try:
        _next_supported(33)
        FAIL("_next_supported raises on >32", "should have raised ValueError")
    except ValueError:
        pass
    PASS("_next_supported")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: RGB-only forward pass (regression)
# ─────────────────────────────────────────────────────────────────────────────

def test_rgb_forward():
    """3-channel render produces output of the right shape and valid range."""
    means3D, scales, rots, opacities, colors = make_gaussians()
    viewmatrix, full_proj, campos, tanfovx, tanfovy, H, W = make_camera()
    bg = torch.zeros(3, device="cuda")
    means2D = torch.zeros_like(means3D, requires_grad=True)

    rasterizer = make_rasterizer(H, W, tanfovx, tanfovy, viewmatrix, full_proj, campos, bg)
    color, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rots,
    )

    assert color.shape  == (3, H, W),    f"color shape wrong: {color.shape}"
    assert allmap.shape == (7, H, W),    f"allmap shape wrong: {allmap.shape}"
    assert radii.shape  == (means3D.shape[0],), f"radii shape wrong: {radii.shape}"
    assert color.min() >= 0.0 - 1e-5,   "color has negative values"
    assert color.max() <= 1.0 + 1e-5,   "color exceeds 1"
    PASS("RGB forward — shapes and value range")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: RGB-only backward (regression)
# ─────────────────────────────────────────────────────────────────────────────

def test_rgb_backward():
    """Backward runs without error; gradient of colors has correct shape."""
    means3D, scales, rots, opacities, colors = make_gaussians()
    colors = colors.requires_grad_(True)
    viewmatrix, full_proj, campos, tanfovx, tanfovy, H, W = make_camera()
    bg = torch.zeros(3, device="cuda")
    means2D = torch.zeros_like(means3D, requires_grad=True)

    rasterizer = make_rasterizer(H, W, tanfovx, tanfovy, viewmatrix, full_proj, campos, bg)
    color, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rots,
    )

    loss = color.mean()
    loss.backward()

    assert colors.grad is not None,              "colors.grad is None"
    assert colors.grad.shape == colors.shape,    f"colors.grad shape: {colors.grad.shape}"
    PASS("RGB backward — grad shapes")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Features forward — various dimensions
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_forward_dims():
    """
    Features of various dimensions (supported and non-supported) produce
    correct output shapes after padding/chunking.
    """
    from diff_surfel_rasterization import rasterization_2dgs

    H, W = 64, 64
    N = 64
    torch.manual_seed(0)
    means   = torch.randn(N, 3, device="cuda"); means[:, 2] += 3.0
    quats   = F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
    scales  = torch.rand(N, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device="cuda"))
    colors  = torch.rand(N, 3, device="cuda")

    fx = fy = W / (2 * math.tan(math.pi / 6))
    Ks = torch.tensor([[[fx, 0, W/2],[0, fy, H/2],[0, 0, 1]]],
                      dtype=torch.float32, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    bg = torch.zeros(1, 3, device="cuda")

    # Test these feature dims: hits supported and non-supported (padded) sizes
    for F_dim in [1, 3, 7, 20, 35]:
        features = torch.rand(N, F_dim, device="cuda")
        result = rasterization_2dgs(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
            backgrounds=bg, features=features,
        )
        render_colors, render_alphas, render_normals, render_normals_from_depth, \
            render_distort, render_median, render_features, info = result

        assert render_colors.shape  == (1, H, W, 3),    f"F={F_dim}: color shape {render_colors.shape}"
        assert render_alphas.shape  == (1, H, W, 1),    f"F={F_dim}: alpha shape {render_alphas.shape}"
        assert render_normals.shape == (1, H, W, 3),    f"F={F_dim}: normals shape {render_normals.shape}"
        assert render_distort.shape == (1, H, W, 1),    f"F={F_dim}: distort shape {render_distort.shape}"
        assert render_median.shape  == (1, H, W, 1),    f"F={F_dim}: median shape {render_median.shape}"
        assert render_features is not None,              f"F={F_dim}: features is None"
        assert render_features.shape == (1, H, W, F_dim), \
            f"F={F_dim}: feature shape {render_features.shape}, expected (1,{H},{W},{F_dim})"

    PASS("feature forward — shapes for dims [1,3,7,20,35]")

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Features don't affect RGB output
# ─────────────────────────────────────────────────────────────────────────────

def test_features_dont_affect_rgb():
    """
    Rendering with and without features should produce identical RGB output,
    since features are rasterized in a separate pass.
    """
    from diff_surfel_rasterization import rasterization_2dgs

    H, W, N = 64, 64, 32
    torch.manual_seed(7)
    means   = torch.randn(N, 3, device="cuda"); means[:, 2] += 3.0
    quats   = F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
    scales  = torch.rand(N, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device="cuda"))
    colors  = torch.rand(N, 3, device="cuda")

    fx = fy = W / (2 * math.tan(math.pi / 6))
    Ks = torch.tensor([[[fx, 0, W/2],[0, fy, H/2],[0, 0, 1]]],
                      dtype=torch.float32, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    bg = torch.zeros(1, 3, device="cuda")

    shared = dict(means=means, quats=quats, scales=scales, opacities=opacities,
                  colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
                  backgrounds=bg)

    rgb_only = rasterization_2dgs(**shared, features=None)
    with_feat = rasterization_2dgs(**shared, features=torch.rand(N, 8, device="cuda"))

    diff = (rgb_only[0] - with_feat[0]).abs().max().item()
    assert diff < 1e-5, f"RGB changed when features added: max diff = {diff}"
    PASS("features don't affect RGB output")

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Detached features — no gradient through rasterizer
# ─────────────────────────────────────────────────────────────────────────────

def test_features_detached():
    """
    Since features are detached, no gradient should flow back to features
    through the rasterizer (consistent with .detach() in the training code).
    """
    from diff_surfel_rasterization import rasterization_2dgs

    H, W, N = 64, 64, 32
    torch.manual_seed(3)
    means   = torch.randn(N, 3, device="cuda"); means[:, 2] += 3.0
    quats   = F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
    scales  = torch.rand(N, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device="cuda"))
    colors  = torch.rand(N, 3, device="cuda")

    # Make features require grad BEFORE detach — simulating the training code
    semantic_source = torch.rand(N, 8, device="cuda", requires_grad=True)
    features = semantic_source.detach()  # matches `semantics.detach()` in training

    fx = fy = W / (2 * math.tan(math.pi / 6))
    Ks = torch.tensor([[[fx, 0, W/2],[0, fy, H/2],[0, 0, 1]]],
                      dtype=torch.float32, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    bg = torch.zeros(1, 3, device="cuda")

    result = rasterization_2dgs(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
        backgrounds=bg, features=features,
    )
    render_colors = result[0]
    render_features = result[6]

    # render_features should NOT be part of the computation graph
    # (features were detached before entering the rasterizer)
    assert not render_features.requires_grad, \
        "render_features.requires_grad=True — detach didn't work"

    # Calling backward on render_colors (which does have a grad_fn via the
    # color rasterization) should NOT propagate gradients to semantic_source,
    # since there is no path through the detached features.
    loss = render_colors.mean()
    loss.backward()

    assert semantic_source.grad is None, \
        "Gradient reached feature source through rasterizer — features were not properly detached"
    PASS("detached features — no gradient to source")

# ─────────────────────────────────────────────────────────────────────────────
# Test 8: rasterization_2dgs return tuple structure
# ─────────────────────────────────────────────────────────────────────────────

def test_return_tuple_structure():
    """
    rasterization_2dgs returns an 8-tuple with the exact same structure
    as gsplat.rasterization_2dgs.
    """
    from diff_surfel_rasterization import rasterization_2dgs

    H, W, N = 64, 64, 32
    torch.manual_seed(1)
    means   = torch.randn(N, 3, device="cuda"); means[:, 2] += 3.0
    quats   = F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
    scales  = torch.rand(N, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device="cuda"))
    colors  = torch.rand(N, 3, device="cuda")

    fx = fy = W / (2 * math.tan(math.pi / 6))
    Ks = torch.tensor([[[fx, 0, W/2],[0, fy, H/2],[0, 0, 1]]],
                      dtype=torch.float32, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    bg = torch.zeros(1, 3, device="cuda")

    result = rasterization_2dgs(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
        backgrounds=bg, features=torch.rand(N, 5, device="cuda"),
    )

    assert len(result) == 8, f"Expected 8-tuple, got {len(result)}"
    render_colors, render_alphas, render_normals, render_normals_from_depth, \
        render_distort, render_median, render_features, info = result

    # Shapes match gsplat convention: [C, H, W, X]
    C = 1
    assert render_colors.shape            == (C, H, W, 3)
    assert render_alphas.shape            == (C, H, W, 1)
    assert render_normals.shape           == (C, H, W, 3)
    assert render_normals_from_depth.shape== (C, H, W, 3)
    assert render_distort.shape           == (C, H, W, 1)
    assert render_median.shape            == (C, H, W, 1)
    assert render_features.shape          == (C, H, W, 5)

    # info dict has required keys
    assert "radii"   in info
    assert "means2d" in info
    assert info["radii"].shape == (C, N)

    PASS("return tuple structure matches gsplat API")

# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Feature chunking — large dimension
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_chunking():
    """
    Feature dim > 32 triggers chunking. Output shape must still be correct
    and the rendered features must be finite (no NaN/Inf from padding issues).
    """
    from diff_surfel_rasterization import rasterization_2dgs

    H, W, N = 64, 64, 32
    torch.manual_seed(5)
    means   = torch.randn(N, 3, device="cuda"); means[:, 2] += 3.0
    quats   = F.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
    scales  = torch.rand(N, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device="cuda"))
    colors  = torch.rand(N, 3, device="cuda")

    fx = fy = W / (2 * math.tan(math.pi / 6))
    Ks = torch.tensor([[[fx, 0, W/2],[0, fy, H/2],[0, 0, 1]]],
                      dtype=torch.float32, device="cuda")
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
    bg = torch.zeros(1, 3, device="cuda")

    for F_dim in [33, 50, 64, 100]:
        features = torch.rand(N, F_dim, device="cuda")
        result = rasterization_2dgs(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmats, Ks=Ks, width=W, height=H,
            backgrounds=bg, features=features,
        )
        render_features = result[6]
        assert render_features is not None,                     f"F={F_dim}: None features"
        assert render_features.shape == (1, H, W, F_dim),      f"F={F_dim}: wrong shape {render_features.shape}"
        assert torch.isfinite(render_features).all(),           f"F={F_dim}: NaN/Inf in features"

    PASS("feature chunking — large dims [33,50,64,100]")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_import,
    test_next_supported,
    test_rgb_forward,
    test_rgb_backward,
    test_feature_forward_dims,
    test_features_dont_affect_rgb,
    test_features_detached,
    test_return_tuple_structure,
    test_feature_chunking,
]

if __name__ == "__main__":
    print("\n=== diff-surfel-rasterization (Object-GS) test suite ===\n")
    passed = 0
    failed = 0
    for test in TESTS:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗  {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(TESTS)} total")
    print(f"{'='*50}\n")
