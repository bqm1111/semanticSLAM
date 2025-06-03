# Project structure
- semantic_cloud: Contain code to generate semantic pointcloud
- semantic_voxblox: Based on Kimera-semantic, taking semantic pointcloud as input and generate semantic voxelized voxblox map
- semantic_slam: contain all launch files and params file
# Run
The demo launch file runs on collected bag files. Change use_sim to false if you want to deploy on a real robot
```bash
roslaunch semantic_slam rmf_semantic_voxblox.launch
```