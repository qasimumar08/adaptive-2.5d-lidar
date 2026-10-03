# Sample Lidar Data

This directory can hold sample Lidar point cloud files for running offline demos and evaluation.

### Supported Formats
- SemanticKITTI binary point cloud files (`*.bin`), containing float32 `[x, y, z, remission]`.
- Corresponding SemanticKITTI label files (`*.label`), containing uint32 labels.

### Usage
When `demo/run_demo.py` is executed without local point cloud files, it automatically falls back to an integrated dynamic driving sequence simulator generating realistic road, terrain, obstacle, vehicle, and pedestrian point clouds.

To test on real-world KITTI or SemanticKITTI sequences, copy sequence frames (e.g. `000000.bin`) into this directory.
