# CloudMetrics
Suite to compare two point clouds geometrically and semantically.

## TUM Data Innovation Lab 2023 - Synthetic Point Cloud Generation
This suite is developped for the TUM Data Innovation Lab 2023 to compare synthetically generated point clouds from a city/road environment with real, labelled LiDAR scans from the same area.

## Requirements
This suite is only developped/tested on Windows
- Docker Desktop
- Visual Studio Code

## How to use
Clone the repository and **change your docker mount path** in the devcontainer.json file. It is important that your .las files lie under the path you are mounting. Then build as a devcontainer in VSCode. Use the jupyter notebook for manipulations and plotting of your labelled .las point clouds.

Currently, operations like computing the Convex Hull and Haussdorff Metric are too intensive to be performed in the notebook. Instead, there are two scripts, one for computing the convex hulls of the two point clouds, a second one for computing the haussdrof Distance using these hulls.

To run the hull compution script, open a terminal and run
```bash
cd src
python3 compute_convex_hulls.py <path/to/pc_1> <path/to/pc_2>
```
This saves the two hulls as .pkl files to your disc into the respective point cloud paths.

Afterwards you can run
```bash
cd src
python3 compute_haussdorff.py <path/to/pc_1> <path/to/pc_2> <path/to/hull_1> <path/to/hull_2> 
```
Depending on your point cloud size, these scripts can take a long time to compute or crash even.
