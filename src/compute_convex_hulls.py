#!/usr/bin/env python
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pickle
from tqdm import tqdm
import os, sys
import helper, params

def partitioned_hull_calculation(xyz, save_path, num_partitions=1000):
    """
    Performs convex hull calculation on a point cloud by partitioning it into smaller subsets.

    Args:
        xyz (array): Input 3D point cloud coordinates as an array indexable by ['x'],['y'] and ['z'].
        num_partitions (int): Number of partitions to divide the point cloud into (default: 1000).
        save_path (str): Path to save the serialized convex hull object.

    Returns:
        True if there was no crash

    Notes:
        - This function divides the point cloud into smaller partitions and calculates the convex hull incrementally.
        - The resulting convex hull object is serialized and saved at the specified save_path.
        - The calculation may take minutes or hours depending on number of point
    """
    convex_hull = None
    partition_size = len(xyz) // num_partitions
    for i in tqdm(range(num_partitions)):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        partition = np.column_stack((xyz[start_idx:end_idx]['x'],
                                    xyz[start_idx:end_idx]['y'],
                                    xyz[start_idx:end_idx]['z']))

        if convex_hull is None:
            convex_hull = ConvexHull(partition, incremental=True)
        else:
            convex_hull.add_points(partition, restart=False)

    # Close the convex hull computation
    convex_hull.close()
    
    with open(save_path, "wb") as file:
        pickle.dump(convex_hull, file)
    return (convex_hull != None)


def compute_hulls(path_pc_real, path_pc_synth):
    """
    Computes convex hulls for two point clouds and saves the hull data to files.

    Args:
        path_pc1 (str): Path to the first point cloud file.
        path_pc2 (str): Path to the second point cloud file.

    Notes:
        - This function imports and prepares the point clouds (class filtering, cropping), extracts the 3D point coordinates, and computes the convex hulls.
        - The convex hull data for each point cloud is saved to separate pickle files.
    """
    cropped_points_1, cropped_points_2 = helper.import_and_prepare_point_clouds(path_pc_real, path_pc_synth, crop=True)
    
    # extract purely the coordinates of the points
    cropped_xyz_1 = cropped_points_1[['x', 'y', 'z']]
    cropped_xyz_2 = cropped_points_2[['x', 'y', 'z']]

    filename_1, _ = os.path.splitext(path_pc_real)
    convex_hull_path_1 = f"{filename_1}_convex_hull.pkl"

    filename_2, _ = os.path.splitext(path_pc_synth)
    convex_hull_path_2 = f"{filename_2}_convex_hull.pkl"

    # Switch to turn off long computation for debugging/testing of other code parts
    RUN = True

    if RUN:
        num_partitions = 1000
        print("Computing ConvexHull for Cloud 1...")
        success = False
        success = partitioned_hull_calculation(cropped_xyz_1, convex_hull_path_1, num_partitions)
        if success:
            print(f"ConvexHull 1 data saved to: {convex_hull_path_1}")
        else:
            print("Cloud 1 Hull computation went wrong")

        print("Computing ConvexHull for synth_xyz_cropped...")
        success = False
        success = partitioned_hull_calculation(cropped_xyz_2, convex_hull_path_2, num_partitions)
        if success:
            print(f"ConvexHull 2 data saved to: {convex_hull_path_2}")
        else:
            print("Cloud 2 Hull computation went wrong")

    return


if __name__ == "__main__":
    if len(sys.argv) < 3:
        path_pc_real = params.DEFAULT_REAL_PC_PATH
        path_pc_synth = params.DEFAULT_SYNTH_PC_PATH
    else:
        path_pc_real = sys.argv[1]
        path_pc_synth = sys.argv[2]
    
    compute_hulls(path_pc_real=path_pc_real, path_pc_synth=path_pc_synth)