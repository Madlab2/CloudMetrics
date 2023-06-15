#!/usr/bin/env python
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
import pickle
from tqdm import tqdm
import os, sys
import helper

def hausdorff_distance(path_pc_1, path_pc_2, path_hull_1, path_hull_2):
    """
    Calculates the Hausdorff distance between two point clouds.

    Args:
        path_pc_1 (str): Path to the first point cloud las file.
        path_pc_2 (str): Path to the second point cloud las file.
        path_hull_1 (str): Path to the file containing the convex hull of the first point cloud (precomputed).
        path_hull_2 (str): Path to the file containing the convex hull of the second point cloud (precomputed).

    Returns:
        float: The Hausdorff distance between the two point clouds.
    """
    cropped_points_1, cropped_points_2 = helper.import_and_prepare_point_clouds(path_pc_1, path_pc_2)
    
    # extract purely the coordinates of the points
    cropped_xyz_1 = cropped_points_1[['x', 'y', 'z']]
    cropped_xyz_2 = cropped_points_2[['x', 'y', 'z']]

    # load hulls
    with open(path_hull_1, "rb") as file:
        hull_1 = pickle.load(file)

    with open(path_hull_2, "rb") as file:
        hull_2 = pickle.load(file)

    # this takes a lot of time
    print("Computing Distance 1...")
    distance1 = directed_hausdorff(cropped_xyz_1, cropped_xyz_2[hull_2.vertices])[0]
    print("Computing Distance 2...")
    distance2 = directed_hausdorff(cropped_xyz_2, cropped_xyz_1[hull_1.vertices])[0]
    print("Done.")

    return max(distance1, distance2)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        path_pc_1 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
        path_pc_2 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
        path_hull_1 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_real.pkl'
        path_hull_2 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_real.pkl'
    else:
        path_pc_1 = sys.argv[1]
        path_pc_2 = sys.argv[2]
        path_hull_1 = sys.argv[3]
        path_hull_2 = sys.argv[4]
    
    d_haussd = hausdorff_distance(path_pc_1=path_pc_1, path_pc_2=path_pc_2,
                        path_hull_1=path_hull_1, path_hull_2=path_hull_2)
    print(d_haussd)