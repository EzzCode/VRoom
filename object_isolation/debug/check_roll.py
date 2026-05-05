"""Render ObjectGS at cam17 actual pose vs look_at pose; save comparison."""
import json, sys, math, numpy as np, cv2
from pathlib import Path

ROOT = r"d:\Engineering\CUFE\GP2\VRoom"
MODEL = r"d:\Engineering\CUFE\GP2\VRoom\temp_deps\ObjectGS\outputs\replica\2d_crossentropy_loss_0.1\office_0\2026-04-28_00-25-22"
OUT = Path(r"d:\Engineering\CUFE\GP2\VRoom\object_isolation\outputs\obj_9\debug_phase05\roll_check")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, ROOT)

from object_isolation.core.object_scope import discover_object_scope
from object_isolation.core.coordinate_frames import look_at_w2c
from target_replenishment.core.objectgs_bridge import render_view, create_virtual_camera

OBJECT_ID = 9
scope, world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(MODEL, OBJECT_ID)

cams = json.load(open(Path(MODEL) / "cameras.json"))
cam = next(c for c in cams if c.get("id") == 17 or c.get("img_name") in ("17", "train_rgb_0680"))
R_train = np.asarray(cam["rotation"], dtype=np.float64)
T_train = np.asarray(cam["position"], dtype=np.float64)
C_W = -R_train.T @ T_train
print("cam17 C_W:", C_W, "img:", cam.get("img_name"))

res = 576
fov_y = math.radians(50.0)
fy = 0.5 * res / math.tan(0.5 * fov_y)
K = np.array([[fy, 0, res/2.0], [0, fy, res/2.0], [0, 0, 1]], dtype=np.float32)

# 1) render at the actual training pose
R_w2c_train = R_train.astype(np.float32)
T_w2c_train = T_train.astype(np.float32)
import torch
bg = torch.ones(3, dtype=torch.float32, device="cuda")

cam_train = create_virtual_camera(R_w2c_train, T_w2c_train, K, res, res)
out_train = render_view(gaussians, cam_train, pipe_config, bg, object_label_id=scope.object_label_id)
rgb_train = (out_train["rgb"].detach().cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)

# 2) render at look_at(C_W, centroid, up_W)
R_la, T_la = look_at_w2c(C_W, world_local.centroid_W, world_local.up_W)
cam_la = create_virtual_camera(R_la, T_la, K, res, res)
out_la = render_view(gaussians, cam_la, pipe_config, bg, object_label_id=scope.object_label_id)
rgb_la = (out_la["rgb"].detach().cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)

# 3) render at look_at with cam17's own up
cam17_up = -R_train[1]
R_la2, T_la2 = look_at_w2c(C_W, world_local.centroid_W, cam17_up)
cam_la2 = create_virtual_camera(R_la2, T_la2, K, res, res)
out_la2 = render_view(gaussians, cam_la2, pipe_config, bg, object_label_id=scope.object_label_id)
rgb_la2 = (out_la2["rgb"].detach().cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)

stack = np.concatenate([rgb_train, rgb_la, rgb_la2], axis=1)
cv2.imwrite(str(OUT / "compare.png"), cv2.cvtColor(stack, cv2.COLOR_RGB2BGR))
print("saved", OUT / "compare.png")
print("left=actual_train_pose | mid=look_at(scope.up_W) | right=look_at(cam17_up)")
