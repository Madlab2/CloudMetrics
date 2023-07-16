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

For calculating the distances and the metric, you can run a script. Change the offsets in `src/params.py` for the real and synthetic point clouds such that they are in the same reference frame. Note the order of operations: First the flip is applied, then the offsets. There is also the option to flip a point cloud along the Y-Axis. The default values for the shifts/flips in this project are

```python
FLIP_REAL = False
X_OFFSET_REAL = -674000
Y_OFFSET_REAL = -5405000
Z_OFFSET_REAL = 0

FLIP_SYNTH = True
ALIGN_SYNTH = True
X_OFFSET_SYNTH = -0.3
Y_OFFSET_SYNTH = -1.4
Z_OFFSET_SYNTH = -1.8
```
The boolean `ALIGN_SYNTH` determines whether the offsets are applied to the synthetic point cloud or not.

To conduct the compairson, run
```bash
cd src
python3 metric.py <path/to/pc_real> <path/to/pc_synthetic> 
```
Depending on the point cloud sizes, this can take a few minutes. The follwoing operations are performed:
1. Transformation of both point clouds to the same frame.
2. Grid-Filtering the point clouds. The parameter `GRID_FILTER_SIZE` can be set in `src/params.py`. The default value is 5 meters.
3. Classwise M3C2 distance calculation. The relevant classes are defined in `src/classes.py`. Classes are weighed by weights defined in `src/params.py`.
4. Cloud-to-Cloud distance calculation. Takes the entire point clouds, i.e. all classes.
5. MeanIoU calculation. Again, classes are defined in `src/classes.py`. Classes are weighed by weights defined in `src/params.py`. 
6. Cloud Distance is computed as a weighted average between MeanM3C2 and Cloud-to-Cloud results. The weigths are `DISTANCE_WEIGHTS` as defined in `src/params.py`.
7. Cloud Metric is computed using a bounded growth function in which a weighted average of MeanM3C2, Cloud-to-Cloud and an inverted MeanIoU factor are inserted. The weigths are `METRIC_WEIGHTS` as defined in `src/params.py`. The rate of the growth function is `SLOPE_FACTOR` as defined in `src/params.py`.


## Other Tools
There is also functionality to compute Hausdorff Distances and convex hulls. 

To run the hull compution script, open a terminal and run
```bash
cd src
python3 compute_convex_hulls.py <path/to/pc_1> <path/to/pc_2>
```
This saves the two hulls as .pkl files to your disc into the respective point cloud paths.

For Haussdorff Distance, you can run
```bash
cd src
python3 compute_haussdorff.py <path/to/pc_1> <path/to/pc_2> <path/to/hull_1> <path/to/hull_2> 
```
Depending on your point cloud size, the hull script can take a long time to compute or even crash. \
For the Haussdorf script, the hull path arguments are optional.
