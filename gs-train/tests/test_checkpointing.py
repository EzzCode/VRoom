import sys
import os
import torch
import torch.nn as nn
from argparse import Namespace
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vroom_core.models.decoder import AppearanceTable as Embedding
from vroom_core.models.gaussian_model import GaussianModel

def test_checkpointing():
    # Setup model
    n_offsets = 5
    feat_dim = 32
    m = GaussianModel(n_offsets=n_offsets, feat_dim=feat_dim, appearance_dim=10)
    
    # Needs some geometry to save
    N = 10
    m._anchor = nn.Parameter(torch.rand(N, 3).cuda())
    m._offset = nn.Parameter(torch.rand(N, n_offsets, 3).cuda())
    m._anchor_feat = nn.Parameter(torch.rand(N, feat_dim).cuda())
    m._scaling = nn.Parameter(torch.rand(N, 6).cuda())
    m._rotation = nn.Parameter(torch.rand(N, 4).cuda())
    m.label_ids = torch.randint(0, 5, (N, 1)).cuda()
    
    # Initialize Embedding so it doesn't fail
    m.embedding_appearance = Embedding(100, 10).cuda()
    # NOTE: do NOT call build_nn_modules() after .cuda() — it recreates the
    # nn.Embedding on CPU, which causes a device mismatch during JIT trace.
    
    # Test path
    test_dir = "/tmp/test_checkpoint_vroom"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    ply_path = os.path.join(test_dir, "point_cloud.ply")
    mlp_dir = os.path.join(test_dir, "mlps")
    
    print("Testing save_ply...")
    m.save_ply(ply_path)
    print("Testing save_mlp_checkpoints...")
    m.save_mlp_checkpoints(mlp_dir)
    
    # Create a new empty model and load
    print("Testing load_ply...")
    m2 = GaussianModel(n_offsets=n_offsets, feat_dim=feat_dim, appearance_dim=10)
    m2.load_ply(ply_path)
    
    print("Testing load_mlp_checkpoints...")
    m2.load_mlp_checkpoints(mlp_dir)
    
    assert m2._anchor.shape == m._anchor.shape
    assert m2._offset.shape == m._offset.shape
    assert m2._anchor_feat.shape == m._anchor_feat.shape
    assert m2._scaling.shape == m._scaling.shape
    assert m2._rotation.shape == m._rotation.shape
    
    # Clean up
    shutil.rmtree(test_dir)
    print("ALL TESTS PASSED SUCCESSFULLY.")


def test_explicit_checkpointing():
    n_offsets = 3
    feat_dim = 8
    model = GaussianModel(n_offsets=n_offsets, feat_dim=feat_dim, appearance_dim=0, gs_attr="2D")

    N = 6
    model._anchor = nn.Parameter(torch.rand(N, 3).cuda())
    model._offset = nn.Parameter(torch.rand(N, n_offsets, 3).cuda())
    model._anchor_feat = nn.Parameter(torch.rand(N, feat_dim).cuda())
    model._scaling = nn.Parameter(torch.rand(N, 6).cuda())
    model._rotation = nn.Parameter(torch.rand(N, 4).cuda())

    test_dir = "/tmp/test_explicit_checkpoint_vroom"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    explicit_path = os.path.join(test_dir, "point_cloud_explicit.ply")
    model.save_explicit(explicit_path)

    loaded = GaussianModel(n_offsets=n_offsets, feat_dim=feat_dim, appearance_dim=0, gs_attr="2D")
    loaded.load_explicit(explicit_path)

    assert loaded.explicit_gs is True
    assert loaded._xyz.shape[0] > 0
    assert loaded._explicit_rgb.shape[1] == 3
    assert loaded._scaling.shape[1] == 3
    assert loaded._rotation.shape[1] == 4

    shutil.rmtree(test_dir)
    print("EXPLICIT CHECKPOINT TEST PASSED.")

if __name__ == "__main__":
    test_checkpointing()
    test_explicit_checkpointing()
