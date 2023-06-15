# CloudMetrics
Suite to compare two point clouds geometrically and semantically.

## TUM Data Innovation Lab 2023 - Synthetic Point Cloud Generation
This suite is developped for the TUM Data Innovation Lab 2023 to compare synthetically generated point clouds from a city/road environment with real, labelled LiDAR scans from the same area.

## Requirements
This suite is only developped/tested on Windows
- Docker Desktop
- Visual Studio Code

## How to use
Clone the repository and change your docker mount in the devcontainer.json file. Then build as a devcontainer in VSCode. Use the jupyter notebook for manipulations and plotting of your labelled .las point clouds. \
Currently, operations like ConvexHull and Haussdorff Metric computation are too intensive to be performed in the notebook. To run a dedicated script, open a terminal and run
```bash
cd src
python3 test_hull_script.py
```
Depending on your point cloud siz, this can take a long time. Note that you need to adapt input and file saving paths in the script manually beforehand.
