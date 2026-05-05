"""Compare cond-frame ObjectGS renders against SV3D cond.

Panels (left→right):
 1) phase-4 conditioning RGBA fed into SV3D  (cam17 GT)
 2) SV3D output frame #20 (conditioning frame, from sv3d_raw cache)
 3) ObjectGS rendered at cam17's actual training R/T (no overrides)
 4) ObjectGS rendered at our orbit pose for cond az/el (current pipeline)
"""
import json, sys, math, numpy as np, cv2, torch
from pathlib import Path

ROOT = r"d:\Engineering\CUFE\GP2\VRoom"
MODEL = r"d:\Engineering\CUFE\GP2\VRoom\temp_deps\ObjectGS\outputs\replica\2d_crossentropy_loss_0.1\office_0\2026-04-28_00-25-22"
OUT = Path(r"d:\Engineering\CUFE\GP2\VRoom\object_isolation\outputs\obj_9\debug_phase05\cond_check")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, ROOT)

from object_isolation.core.object_scope import discover_object_scope
from object_isolation.core.coordinate_frames import look_at_w2c
from target_replenishment.core.objectgs_bridge import render_view, create_virtual_camera

OBJECT_ID = 9
RES = 576
FOV_Y = 50.0

scope, world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(MODEL, OBJECT_ID)

# 1) Phase-4 cond input
cond_in = cv2.imread(r"d:\Engineering\CUFE\GP2\VRoom\object_isolation\outputs\obj_9\phase5\conditioning.png")
cond_in = cv2.resize(cond_in, (RES, RES))

# 2) SV3D frame #20 (cond) from cache
sv3d_dir = Path(r"d:\Engineering\CUFE\GP2\VRoom\object_isolation\outputs\obj_9\phase5\sv3d_raw")
sv3d_files = sorted(sv3d_dir.glob("*.png"))
print("sv3d files:", [f.name for f in sv3d_files[:3]], "...", [f.name for f in sv3d_files[-2:]])
sv3d_cond = cv2.imread(str(sv3d_files[-1]))
sv3d_cond = cv2.resize(sv3d_cond, (RES, RES))

# Build K
fy = 0.5 * RES / math.tan(0.5 * math.radians(FOV_Y))
K = np.array([[fy, 0, RES/2.0], [0, fy, RES/2.0], [0, 0, 1]], dtype=np.float32)

bg = torch.ones(3, dtype=torch.float32, device="cuda")

# 3) Render at cam17's actual training pose
cam17 = scope.cameras[17]
R_train = cam17["R"].astype(np.float32)
T_train = cam17["T"].astype(np.float32)
print("cam17 C_W:", -R_train.T @ T_train)
print("cam17 dist to centroid:", np.linalg.norm(-R_train.T @ T_train - scope.centroid_W))
# Use cam17's NATIVE intrinsics + resolution to reproduce its actual view
K_native = cam17["K"].astype(np.float32)
W_n, H_n = cam17["width"], cam17["height"]
print(f"cam17 native: {W_n}x{H_n} fx={K_native[0,0]:.1f} fy={K_native[1,1]:.1f}")
cam_actual = create_virtual_camera(R_train, T_train, K_native, W_n, H_n)
out_actual = render_view(gaussians, cam_actual, pipe_config, bg, object_label_id=scope.object_label_id)
rgb_actual = (out_actual["rgb"].detach().cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)
rgb_actual = cv2.resize(rgb_actual, (RES, RES))

# 4) Render at our orbit pose for cond az/el
sc = json.load(open(r"d:\Engineering\CUFE\GP2\VRoom\object_isolation\outputs\obj_9\phase4\scores.json"))
top1 = sc["top_k"][0]
full = next(fr for fr in sc["frames"] if fr["cam_index"] == top1["cam_index"])
cond_az = float(full["azimuth_V_deg"])
cond_el = float(full["elevation_V_deg"])
print(f"cond az_V={cond_az:.2f} el_V={cond_el:.2f}")

R_orb, T_orb, C_orb = local_sv3d.sv3d_view_to_world_camera(cond_az, cond_el)
print("orbit C_W:", C_orb, "dist:", np.linalg.norm(C_orb - scope.centroid_W))
# Apply cond up override (current code path)
cond_up = -cam17["R"][1].astype(np.float64)
cond_up /= np.linalg.norm(cond_up)
R_orb_up, T_orb_up = look_at_w2c(C_orb, scope.centroid_W, cond_up)
cam_orb = create_virtual_camera(R_orb_up.astype(np.float32), T_orb_up.astype(np.float32), K, RES, RES)
out_orb = render_view(gaussians, cam_orb, pipe_config, bg, object_label_id=scope.object_label_id)
rgb_orb = (out_orb["rgb"].detach().cpu().numpy().transpose(1,2,0) * 255).clip(0,255).astype(np.uint8)

# Convert RGB renders to BGR for cv2.imwrite
rgb_actual_bgr = cv2.cvtColor(rgb_actual, cv2.COLOR_RGB2BGR)
rgb_orb_bgr = cv2.cvtColor(rgb_orb, cv2.COLOR_RGB2BGR)
stack = np.concatenate([cond_in, sv3d_cond, rgb_actual_bgr, rgb_orb_bgr], axis=1)

# Add labels
labels = ["1) cond input (cam17 GT)", "2) SV3D #20 (cond)", "3) ObjGS @ cam17 actual R/T", "4) ObjGS @ orbit pose (cond up)"]
for i, lbl in enumerate(labels):
    cv2.putText(stack, lbl, (i * RES + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite(str(OUT / "compare.png"), stack)
print("saved", OUT / "compare.png")
