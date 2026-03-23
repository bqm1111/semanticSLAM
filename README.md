# Semantic SLAM

A ROS-based semantic 3D mapping system that fuses real-time semantic segmentation with volumetric TSDF mapping. The system uses a diffusion-based RGB-D semantic segmentation model to label point clouds with semantic class information, then integrates them into a 3D semantic voxel map using a modified Voxblox TSDF integrator with Bayesian label updates.

## System Overview

```
RealSense D455 (RGB + Depth)
        │
        ▼
 [semantic_cloud]
  DiffusionMMS segmentation
  Depth-to-color alignment
  Semantic point cloud generation
        │
        ├── /semantic_pcl/semantic_pcl  (PointCloud2 with RGB-encoded labels)
        └── /semantic_pcl/camera_pose  (TransformStamped)
                │
                ▼
        [semantic_voxblox]
         Bayesian semantic TSDF integration
         3D semantic mesh + ESDF generation
```

Odometry is provided by ROVIO (`/rovio/odometry`), fusing IMU and visual data.

## Packages

| Package | Language | Description |
|---|---|---|
| `semantic_cloud` | Python | RGB-D semantic segmentation → labeled PointCloud2 |
| `semantic_voxblox` | C++ | Semantic TSDF volumetric mapping (derived from Voxblox / Kimera-Semantics) |
| `align_depth_to_color` | Python | Utility node: align depth images to color camera frame |
| `octomap_generator` | C++ | OctoMap-based semantic mapping (legacy, currently disabled) |
| `semantic_slam` | — | Meta-package: launch files and configuration parameters |

## Dependencies

### System / ROS

- ROS Melodic or Noetic
- `cv_bridge`, `tf`, `tf2_ros`, `message_filters`
- `sensor_msgs`, `geometry_msgs`, `nav_msgs`
- `pcl_ros`, `pcl_conversions`

### External ROS packages

- [voxblox](https://github.com/ethz-asl/voxblox) (ETH ASL) — `voxblox`, `voxblox_ros`, `voxblox_msgs`, `voxblox_rviz_plugin`
- [catkin_simple](https://github.com/catkin/catkin_simple)
- `minkindr_conversions`, `gflags_catkin`, `glog_catkin`
- ROVIO (for odometry)
- ORB-SLAM3 + `hector_trajectory_server` (TUM RGB-D launch only)

### Python

- PyTorch with CUDA
- [DiffusionMMS](https://github.com/ntnu-arl/diffusionMMS) (git submodule at `semantic_cloud/include/diffusionMMS`)
- `numpy`, `opencv-python`, `numba`, `omegaconf`

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
```

Or, if already cloned:

```bash
git submodule update --init --recursive
```

### 2. Install Python dependencies

```bash
pip install torch torchvision numpy opencv-python numba omegaconf
```

Install the DiffusionMMS package:

```bash
pip install -e semantic_cloud/include/diffusionMMS
```

### 3. Build

```bash
cd <catkin_ws>
catkin build semantic_voxblox semantic_cloud semantic_slam align_depth_to_color
```

## Configuration

### Main parameters (`semantic_slam/params/rmf.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `color_image_topic` | `/d455/color/image_raw` | RGB image topic |
| `depth_image_topic` | `/d455/depth/image_rect_raw` | Depth image topic |
| `color_cam_info_topic` | `/d455/color/camera_info` | Color camera info topic |
| `depth_cam_info_topic` | `/d455/depth/camera_info` | Depth camera info topic |
| `camera_pose_topic` | `/rovio/odometry` | Odometry source |
| `enable_semantic` | `True` | Enable DiffusionMMS segmentation |
| `process_sematic_freq` | `10` | Process every N-th frame |
| `depth_scale` | `0.001` | Depth image scale factor (mm → m) |
| `selected_semantic` | *(38 NYUv2 classes)* | Classes to include in the point cloud |

### Voxblox parameters (`semantic_slam/params/voxblox_cfg.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `tsdf_voxel_size` | `0.03` | Voxel resolution (3 cm) |
| `tsdf_voxels_per_side` | `16` | Block size |
| `max_ray_length_m` | `5.0` | Maximum integration depth |
| `method` | `fast` | TSDF integrator type (`fast` or `merged`) |
| `semantic_measurement_probability` | `0.8` | Bayesian update confidence per observation |

### Model checkpoint

Set in launch file or override at runtime:

```xml
<arg name="semseg_model_ckpt" default="<path_to_checkpoint>.pth" />
<arg name="semseg_model_cfg"  default="<path_to_config>.yaml" />
```

## Launch

### RMF setup (RealSense D455 + ROVIO)

```bash
roslaunch semantic_slam rmf_semantic_voxblox.launch
```

Optionally override model paths:

```bash
roslaunch semantic_slam rmf_semantic_voxblox.launch \
  semseg_model_ckpt:=/path/to/checkpoint.pth \
  semseg_model_cfg:=/path/to/config.yaml
```

### TUM RGB-D dataset (with ORB-SLAM3)

```bash
roslaunch semantic_slam tum_rgbd.launch bag_file:=/path/to/dataset.bag
```

### Demo (rosbag playback)

```bash
roslaunch semantic_slam semantic_mapping.launch bag_file:=/path/to/demo.bag
```

### Depth alignment utility

```bash
roslaunch align_depth_to_color align_depth_to_color.launch
```

## ROS Interface

### Published topics

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/semantic_pcl/semantic_pcl` | `sensor_msgs/PointCloud2` | semantic_cloud | Semantic point cloud (XYZ + RGB-encoded label colors) |
| `/semantic_pcl/semantic_image` | `sensor_msgs/Image` | semantic_cloud | Colorized segmentation visualization |
| `/semantic_pcl/camera_pose` | `geometry_msgs/TransformStamped` | semantic_cloud | Camera pose in world frame |
| `/tsdf_pointcloud` | `sensor_msgs/PointCloud2` | semantic_voxblox | TSDF map as point cloud |
| `/surface_mesh` | `visualization_msgs/MarkerArray` | semantic_voxblox | Semantic mesh for RViz |

### Subscribed topics (semantic_cloud)

| Topic | Type | Description |
|---|---|---|
| `/d455/color/image_raw` | `sensor_msgs/Image` | RGB image |
| `/d455/depth/image_rect_raw` | `sensor_msgs/Image` | 16-bit depth image |
| `/d455/color/camera_info` | `sensor_msgs/CameraInfo` | Color intrinsics |
| `/d455/depth/camera_info` | `sensor_msgs/CameraInfo` | Depth intrinsics |
| `/rovio/odometry` | `nav_msgs/Odometry` | Camera odometry from ROVIO |

## Semantic Classes

The system uses the **NYUv2 40-class** label set:

`wall`, `floor`, `cabinet`, `bed`, `chair`, `sofa`, `table`, `door`, `window`, `bookshelf`, `picture`, `counter`, `blinds`, `desk`, `shelves`, `curtain`, `dresser`, `pillow`, `mirror`, `floor mat`, `clothes`, `ceiling`, `books`, `refridgerator`, `television`, `paper`, `towel`, `shower curtain`, `box`, `whiteboard`, `person`, `night stand`, `toilet`, `sink`, `lamp`, `bathtub`, `bag`, `otherstructure`, `otherfurniture`, `otherprop`

Label-to-color mappings are defined in `semantic_slam/params/nyu.csv`.

## TF Frame Tree

```
world
└── map (= mocap)
    └── imu
        └── base_link
            └── camera0  (= d455_depth_optical_frame)
                └── d455_color_optical_frame
```

Static transforms are published by `semantic_slam/launch/tf_static.launch`.

## Credits

- **semantic_voxblox**: Derived from [voxblox](https://github.com/ethz-asl/voxblox) (ETH ASL, BSD) and [Kimera-Semantics](https://github.com/MIT-SPARK/Kimera-Semantics) (MIT, Antoni Rosinol)
- **DiffusionMMS**: Diffusion-based multimodal semantic segmentation model ([ntnu-arl/diffusionMMS](https://github.com/ntnu-arl/diffusionMMS))
