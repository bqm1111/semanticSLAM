# Adapting a New Segmentation Model

This guide explains how to replace the DiffusionMMS segmentation model with your own model while keeping the rest of the semantic mapping pipeline unchanged.

---

## How the Pipeline Works

Understanding the data flow helps you know exactly what needs to change.

```
color_depth_callback()
    │
    ├─ predict(model, color_img, aligned_depth)
    │       └─ returns: tensor (1, H, W) of integer class indices
    │
    ├─ colorized_prediction(semantic_pred, num_classes)
    │       └─ returns: (colored_img HxWx3, class_index_array HxW, valid_mask HxW bool)
    │
    └─ build PointCloud2 with RGB-encoded label colors
              └─ published to /semantic_pcl/semantic_pcl
                          └─ consumed by semantic_voxblox
```

The semantic_voxblox node receives the point cloud and uses the RGB color of each point to look up its semantic class in the CSV file. It then performs Bayesian label updates in the TSDF voxels.

**Your model only needs to plug into one place: the `predict()` function.**
The rest of the pipeline is model-agnostic.

---

## What You Need to Change

All changes are confined to `semantic_cloud/src/diffusion_cloud.py` plus two configuration files.

| What | Where | Why |
|---|---|---|
| Class name list | `SEMANTIC_CLASSES_NAMES` in `diffusion_cloud.py` | Maps index → name |
| Model loading | `setup_model()` in `diffusion_cloud.py` | Load your model weights |
| Preprocessing | `preprocess()` in `diffusion_cloud.py` | Your model's input format |
| Inference | `predict()` in `diffusion_cloud.py` | Run your model |
| Color assignment | `colorized_prediction()` in `diffusion_cloud.py` | Visualization + label encoding |
| Label-to-color table | `semantic_slam/params/nyu.csv` (copy and edit) | semantic_voxblox color lookup |
| Selected classes | `semantic_slam/params/rmf.yaml` | Filter which classes enter the map |
| Launch file | `semantic_slam/launch/rmf_semantic_voxblox.launch` | Model paths |

---

## Step 1 — Define Your Class Names

At the top of `diffusion_cloud.py`, replace `SEMANTIC_CLASSES_NAMES` with your model's classes **in the exact order your model outputs them** (index 0 = first entry, index 1 = second entry, etc.).

```python
# Before (DiffusionMMS NYUv2 40 classes)
SEMANTIC_CLASSES_NAMES = ["wall", "floor", "cabinet", ...]  # 40 entries

# After (your model — example with 20 classes)
SEMANTIC_CLASSES_NAMES = [
    "background",   # index 0
    "wall",         # index 1
    "floor",        # index 2
    "chair",        # index 3
    # ... one entry per class, in the same order as your model's output
]
```

**Critical:** The index of each name in this list must match the integer your model outputs for that class.

---

## Step 2 — Replace Model Loading

Replace the `setup_model` function with one that loads your model:

```python
# Before (DiffusionMMS specific)
def setup_model(cfg_file, device, ckpt_path):
    config = OmegaConf.load(cfg_file)
    model = get_model(config.model.name, eval=True, **config.model.params)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model = model.to(device)
    return model

# After (your model — adapt as needed)
def setup_model(cfg_file, device, ckpt_path):
    # Example: a simple torchvision-style model
    model = YourModelClass(num_classes=len(SEMANTIC_CLASSES_NAMES))
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)   # adjust key if needed, e.g. checkpoint["model_state_dict"]
    model = model.to(device)
    model.eval()
    return model
```

Remove any imports that were only needed for DiffusionMMS (e.g. `from diffusionMMS.engine import get_model`).

---

## Step 3 — Replace Preprocessing

Replace `preprocess()` with your model's expected input format.

```python
# Before (DiffusionMMS — takes both RGB and depth as input)
def preprocess(input_rgb, input_depth):
    depth = copy.copy(input_depth)
    rgb = copy.copy(input_rgb)
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb / 255.0 - mean) / std
    depth = convert_depth_to_three_channel_img(depth) / 255.0
    rgb = rgb.transpose(2, 0, 1)
    depth = depth.transpose(2, 0, 1)
    rgb = torch.from_numpy(rgb).unsqueeze(0).cuda().float()
    depth = torch.from_numpy(depth).unsqueeze(0).cuda().float()
    return {"rgb": rgb, "depth": depth}
```

**RGB-only model example:**
```python
def preprocess(input_rgb, input_depth):
    rgb = input_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    rgb = (rgb - mean) / std
    rgb = rgb.transpose(2, 0, 1)                        # HxWx3 -> 3xHxW
    rgb = torch.from_numpy(rgb).unsqueeze(0).cuda().float()  # 1x3xHxW
    return {"rgb": rgb}
```

**RGB-D model with depth as 4th channel example:**
```python
def preprocess(input_rgb, input_depth):
    rgb = input_rgb.astype(np.float32) / 255.0
    depth = input_depth.copy()
    depth = np.clip(depth / 10.0, 0, 1)                # normalize to [0, 1] assuming max 10m
    depth[~np.isfinite(depth)] = 0
    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    rgbd = np.concatenate([rgb, depth[:, :, np.newaxis]], axis=2)  # HxWx4
    rgbd = rgbd.transpose(2, 0, 1)
    rgbd = torch.from_numpy(rgbd).unsqueeze(0).cuda().float()
    return {"rgbd": rgbd}
```

---

## Step 4 — Replace the Inference Call

Replace `predict()` to call your model and return a `(1, H, W)` integer tensor of class indices.

**This is the most critical contract.** The output of `predict()` is directly used to:
- Assign colors to each pixel for visualization
- Filter which pixels enter the point cloud (by class name)

```python
# Before (DiffusionMMS — calls model.sampling())
def predict(model, rgb, depth):
    data = preprocess(rgb, depth)
    with torch.no_grad():
        score = model.sampling(data["rgb"], data["depth"])  # returns (1, C, H, W) logits
    pred = score.argmax(1)   # -> (1, H, W) integer class indices
    return pred

# After (standard PyTorch model)
def predict(model, rgb, depth):
    data = preprocess(rgb, depth)
    with torch.no_grad():
        logits = model(data["rgb"])   # adjust keys to match your preprocess output
    pred = logits.argmax(dim=1)       # -> (1, H, W) integer class indices
    return pred
```

**Output contract:** `predict()` must return a tensor with:
- Shape: `(1, H, W)` where H, W match the input image size
- Dtype: integer (e.g. `torch.int64`)
- Values: class indices in range `[0, len(SEMANTIC_CLASSES_NAMES) - 1]`

---

## Step 5 — Update Color Assignment

`colorized_prediction()` does two things: creates a visualization image and encodes the class label into a per-pixel color that semantic_voxblox uses to look up the label in the CSV.

The current version calls `get_class_colors()` from DiffusionMMS. Replace this with your own color map.

```python
# Before (uses DiffusionMMS helper)
def colorized_prediction(pred, num_classes=40):
    colors = np.array(get_class_colors(num_classes + 1))
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    trimmed_mask, valid_mask = erode_segmentation_mask(pred_arr, kernel_size=5, iterations=2)
    colored_pred = np.zeros_like(pred_arr)
    colored_pred = np.stack((colored_pred,) * 3, axis=-1)
    colored_pred[:] = colors[trimmed_mask[:]]
    return colored_pred, np.array(trimmed_mask), valid_mask
```

**After — define your own colors:**

The colors here must exactly match the RGB values in your CSV file (Step 6), because semantic_voxblox identifies classes by matching point cloud colors to the CSV lookup table.

```python
# Define colors to match your CSV file (one RGB tuple per class, index = class id)
MY_CLASS_COLORS = np.array([
    [  0,   0,   0],   # 0: background  -> must match CSV row id=0
    [  0,   0, 128],   # 1: wall        -> must match CSV row id=1
    [  0, 128,   0],   # 2: floor       -> must match CSV row id=2
    [  0, 128, 128],   # 3: chair       -> must match CSV row id=3
    # ... one row per class
], dtype=np.uint8)

def colorized_prediction(pred, num_classes=None):
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    trimmed_mask, valid_mask = erode_segmentation_mask(pred_arr, kernel_size=5, iterations=2)
    colored_pred = MY_CLASS_COLORS[trimmed_mask]   # HxWx3
    return colored_pred, np.array(trimmed_mask), valid_mask
```

Also update the call site in `color_depth_callback` to pass the correct `num_classes`:
```python
semantic_color, pred_arr, eroded_valid_mask = colorized_prediction(semantic_pred, num_classes=len(SEMANTIC_CLASSES_NAMES))
```

---

## Step 6 — Create a Label-to-Color CSV File

semantic_voxblox reads this file to decode which semantic class each colored point belongs to. The RGB values here **must exactly match** those you defined in `MY_CLASS_COLORS` in Step 5.

Copy `semantic_slam/params/nyu.csv` as a starting point:
```bash
cp semantic_slam/params/nyu.csv semantic_slam/params/my_classes.csv
```

Edit `my_classes.csv` to list your classes. The format is:
```
name,red,green,blue,alpha,id
background,0,0,0,255,0
wall,0,0,128,255,1
floor,0,128,0,255,2
chair,0,128,128,255,3
```

Rules:
- `id` is the integer class index (must match your model's output and `SEMANTIC_CLASSES_NAMES` order)
- `red,green,blue` must exactly match the color for that class in `MY_CLASS_COLORS`
- `alpha` can always be `255`

---

## Step 7 — Update the Warm-Up Call

The warm-up in `SemanticCloud.__init__` pre-runs the model once to force Numba/CUDA JIT compilation before real data arrives. Update the dummy input to match your model's expected input shape.

```python
# Before (DiffusionMMS — hardcoded 480x848 RGB-D)
predict(self.model, np.ones((480, 848, 3), dtype=np.uint8), np.random.rand(480, 848))

# After — use your camera's actual resolution
H, W = 480, 640   # replace with your camera height and width
predict(self.model, np.ones((H, W, 3), dtype=np.uint8), np.random.rand(H, W))
```

---

## Step 8 — Update Configuration Files

### `semantic_slam/params/rmf.yaml` — choose which classes to include

Update `selected_semantic` to list only the class names you care about (using names from your new `SEMANTIC_CLASSES_NAMES`). Classes not listed here are excluded from the point cloud.

```yaml
# Before
selected_semantic:
  ["wall", "floor", "chair", "table", ...]   # NYUv2 names

# After — use names from your SEMANTIC_CLASSES_NAMES
selected_semantic:
  ["wall", "floor", "chair"]   # only these will appear in the map
  # Or include everything:
  # selected_semantic: ["all"]
```

### `semantic_slam/launch/rmf_semantic_voxblox.launch` — point to your model and CSV

```xml
<!-- Before -->
<arg name="semseg_model_ckpt"
    default="$(find semantic_cloud)/include/diffusionMMS/output_dir/nyuv2/.../checkpoint-92.pth" />
<arg name="semseg_model_cfg"
    default="$(find semantic_cloud)/include/diffusionMMS/config/nyuv2/.../config.yaml" />
<param name="semantic_label_2_color_csv_filepath" value=".../nyu.csv" />

<!-- After -->
<arg name="semseg_model_ckpt" default="/path/to/your_model.pth" />
<arg name="semseg_model_cfg"  default="/path/to/your_config.yaml" />   <!-- omit if your model doesn't use a config file -->
<param name="semantic_label_2_color_csv_filepath" value="$(find semantic_slam)/params/my_classes.csv" />
```

If your model does not use a config file, remove the `semseg_model_cfg` parameter from the launch file and remove it from `setup_model()`.

---

## Step 9 — Test

**Test 1: Check segmentation output in isolation**

```bash
# Launch only the semantic_cloud node
roslaunch semantic_slam rmf_semantic_voxblox.launch
```

In a second terminal, check that the segmentation image publishes correctly:
```bash
rosrun rqt_image_view rqt_image_view /semantic_pcl/semantic_image
```

You should see a colorized segmentation overlay on the camera feed.

**Test 2: Verify the point cloud colors match the CSV**

```bash
rostopic echo /semantic_pcl/semantic_pcl | head -20
```

The `rgb` field in the point cloud encodes the class color. Verify it matches the expected values from your `MY_CLASS_COLORS`.

**Test 3: Check semantic_voxblox is receiving and integrating**

Look for log output from `semantic_voxblox` after the first point cloud arrives:
```bash
rostopic hz /semantic_pcl/semantic_pcl   # should be ~1–2 Hz depending on process_sematic_freq
```

In RViz, the `/surface_mesh` topic should show a colored 3D mesh.

---

## Common Mistakes

**Colors don't match between Python and CSV**
The exact same RGB triplet must appear both in `MY_CLASS_COLORS[i]` and the CSV row for class `i`. A difference of even 1 in any channel will cause semantic_voxblox to treat the point as an unknown class.

**Class index mismatch**
`SEMANTIC_CLASSES_NAMES[i]` must correspond to class index `i` as your model outputs it. Double-check against your model's documentation or training config.

**`selected_semantic` names don't match `SEMANTIC_CLASSES_NAMES`**
The name filtering in `color_depth_callback` does a direct string lookup into `SEMANTIC_CLASSES_NAMES`. Typos or different casing will silently drop all points for that class.

**Wrong input resolution in warm-up**
If the warm-up dummy tensor has a different shape than real camera images, the Numba JIT cache will recompile on first real frame, causing a delay. Set the warm-up resolution to match your actual camera.

**Model outputs float logits, not integer indices**
The output of `predict()` must be integer class indices (after `argmax`), not raw logits. If you return logits directly, the downstream color lookup and class filtering will produce garbage.
