# Mask Pipeline

## What we are trying to do

`object_refiner` needs to extract **per-object RGBA frames** from a multi-object scene.
For each training camera view we want to know: *which pixels belong to object X and nothing else?*

The answer comes entirely from **masks_and_tracking's per-frame segmentation maps**.

---

## The segmentation map (`seg_mask`)

masks_and_tracking runs a **2D instance segmentation** tracker (e.g. DEVA/SAM) on the raw video frames.
For every input frame it produces a single PNG where **each pixel's value is the integer instance
label** of the object at that pixel (0 = background).

```
frame_0001.png  →  seg map where pixel value 7 = "mug", 3 = "table", 0 = background
```

`vote.py` then takes those per-frame 2D seg maps and projects the COLMAP 3D sparse points
onto each frame. A 3D point gets the label from whichever 2D pixel it lands on. After voting
across all views, every 3D point ends up with a majority-vote label. The output is
`points3D_labeled.ply` — the same COLMAP point cloud but with a `label` field on each point.

**The integer labels in the seg maps and in the labeled PLY are the same numbers.**
If object X has `seg_label = 7`, then all its 2D seg map pixels are 7, and all its 3D points in
the labeled PLY also have label 7.

```python
seg_mask = (seg_map == seg_label)   # binary mask: True where masks_and_tracking labelled this object
```

A missing seg map for a camera is a hard error — the pipeline raises `RuntimeError` rather than
silently producing a bad mask.

---

## Label discovery — `vote_seg_label()`

The seg maps and the ObjectGS model use different namespaces for objects. ObjectGS knows the object
by its `object_label_id` (an integer assigned during ObjectGS training). The seg maps use their own
instance labels assigned by the 2D tracker. These are **not guaranteed to be the same number**.

`vote_seg_label()` in `helpers.py` bridges the gap. It is run **once before extraction**:

1. Renders the GS alpha for a small probe set of cameras using `render_rgba(object_label_id=X)`.
2. For each probe camera, computes IoU between the thresholded GS alpha and every unique label
   in the seg map.
3. Accumulates IoU scores across cameras; the label with the highest total wins.

```python
seg_label = vote_seg_label(scope, gaussians, pipe_config, seg_map_dir)
```

That `seg_label` integer is then passed into every `extract_frame` call.

**This function is temporary.** In production the labels will be aligned at the pipeline level
(ObjectGS training will be seeded with the same label IDs that masks_and_tracking outputs), so `seg_label`
will be known directly and `vote_seg_label` can be deleted.

---

## Data flow

```
masks_and_tracking tracker
    │
    ├─ per-frame seg maps  (PNG, pixel value = instance label)
    │       └─ vote.py projects them  →  points3D_labeled.ply
    │
    │   Both outputs share the same integer label (seg_label)
    │
    │            [temporary — until label alignment is done]
    │            vote_seg_label() renders GS alpha for ~5 cameras
    │            and IoU-votes it against the seg maps to find seg_label
    │
object_refiner extraction  (view_selection.py)
    │
    └─ _load_seg_mask(seg_map_dir, image_name, seg_label)
               → seg_mask is the sole source of per-frame object pixels
               → stored in the RGBA alpha channel of extracted/*.png
```

---

## Code path

| Step | Code | What it does |
|------|------|--------------|
| Label discovery | `vote_seg_label()` in `helpers.py` | finds which seg map label matches this object; runs once before extraction |
| Load seg map | `_load_seg_mask(seg_map_dir, image_name, seg_label, shape)` in `view_selection.py` | reads the per-frame PNG, isolates `seg_label` pixels |
| Hard error on miss | `RuntimeError` in `extract_frame` | seg map must exist for every visible camera |
| Post-process | `_close_and_fill` + `_largest_cc` | fills small holes, drops stray specks |
| Save | written to `extracted/<cam>__<name>.png` as the alpha channel | downstream reads `out_rgba_path[..., 3]` |

---

## Future cleanup

Once label alignment is in place end-to-end, remove:

- `vote_seg_label()` from `helpers.py`
- The `vote_seg_label` call and `gaussians`/`pipe_config` dependency in `__main__.py`
- The `gstrain_wrapper` import from `helpers.py`
