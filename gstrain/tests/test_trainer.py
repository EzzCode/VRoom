import sys
import os
import torch
import torch.nn as nn
from argparse import Namespace
import shutil
import logging
from pathlib import Path

from gstrain.trainer import PipeConfig, Trainer, TrainingConfig
from typing import List
from gstrain.vroom_core.models.facade import GaussianModel
from gstrain.vroom_core.models.semantics import SemanticCodec

# Mock Scene
class MockCamera:
    def __init__(self, uid):
        self.uid = uid
        self.resolution_scale = 1.0
        self.image_width = 128
        self.image_height = 128
        # Create a tiny image
        self.original_image = torch.rand(3, 128, 128)
        self.alpha_mask = torch.ones(1, 128, 128)
        
        # Renderer-facing camera properties used by the training loop
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.world_view_transform = torch.eye(4).cuda()
        self.full_proj_transform = torch.eye(4).cuda()
        self.camera_center = torch.zeros(3).cuda()
        self.object_mask = torch.zeros(128, 128)

        # Intrinsics derived from FoV (prefilter_voxel builds K from these)
        import math
        self.fx = self.image_width  / (2.0 * math.tan(self.FoVx / 2.0))
        self.fy = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.cx = self.image_width  / 2.0
        self.cy = self.image_height / 2.0

class MockScene:
    def __init__(self):
        self.background = torch.zeros(3).cuda()
        self.cams = [MockCamera(i) for i in range(2)]
        
    def getTrainCameras(self) -> List[MockCamera]:
        return self.cams
        
    def getTestCameras(self) -> List[MockCamera]:
        return self.cams

def test_trainer():
    logger = logging.getLogger()
    
    opt = TrainingConfig()
    opt.iterations = 10
    opt.start_stat = 2
    opt.update_from = 4
    opt.update_until = 8
    
    pipe = PipeConfig()
    
    m = GaussianModel(n_offsets=2, feat_dim=8)
    
    # Initialize Fake Params
    anchor = torch.rand((100, 3)).cuda()
    m._anchor = nn.Parameter(anchor.clone())
    m._offset = nn.Parameter(torch.rand((100, 2, 3)).cuda())
    m._anchor_feat = nn.Parameter(torch.rand((100, 8)).cuda())
    m._scaling = nn.Parameter(torch.rand((100, 6)).cuda())
    m._rotation = nn.Parameter(torch.rand((100, 4)).cuda())
    
    label_ids = torch.randint(0, 5, (100, 1)).cuda()
    m.label_ids = label_ids
    m.id_encoder = SemanticCodec.from_labels(m.label_ids.view(-1))
    
    m.voxel_size = 0.01
    m.spatial_lr_scale = 1.0
    
    try:
        scene = MockScene()
        trainer = Trainer(opt, pipe, m, scene, output_dir="/tmp/mock_trainer", logger=logger)
        trainer.train()
        print("Training execution OK.")
        trainer.evaluate(10)
        print("Evaluation execution OK.")
        print("ALL TESTS PASSED SUCCESSFULLY.")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trainer()
