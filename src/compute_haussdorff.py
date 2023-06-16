#!/usr/bin/env python
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
import pickle
from tqdm import tqdm
import concurrent.futures
import os, sys
import helper


def laspy_to_np_array(laspy_points):
    x = laspy_points['x']
    y = laspy_points['y']
    z = laspy_points['z']
    return np.column_stack((x, y, z))

def build_kdtree(points):
    return cKDTree(points)

def hausdorff_distance_with_tree(path_pc_1, path_pc_2, path_hull_1=None, path_hull_2=None):
    """
    Calculates the Hausdorff distance between two point clouds using their convex hulls and spatial data structures.

    Args:
        path_pc_1 (str): Path to the first point cloud las file.
        path_pc_2 (str): Path to the second point cloud las file.
        path_hull_1 (str, optional): Path to the file containing the convex hull of the first point cloud (precomputed).
        path_hull_2 (str, optional): Path to the file containing the convex hull of the second point cloud (precomputed).

    Returns:
        float: The Hausdorff distance between the two point clouds.
    """
    cropped_points_1, cropped_points_2 = helper.import_and_prepare_point_clouds(path_pc_1, path_pc_2)
    
    # extract purely the coordinates of the points
    cropped_xyz_1 = cropped_points_1[['x', 'y', 'z']]
    cropped_xyz_2 = cropped_points_2[['x', 'y', 'z']]
    
    # convert to np.array. Crucial for kd tree construction performance
    cropped_xyz_1 = laspy_to_np_array(cropped_xyz_1)
    cropped_xyz_2 = laspy_to_np_array(cropped_xyz_2)

    print("Building kd-tree for the first point cloud...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tree_1 = executor.submit(build_kdtree, cropped_xyz_1).result()
    print("Building kd-tree for the second point cloud...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tree_2 = executor.submit(build_kdtree, cropped_xyz_2).result()
    
    
    # Different computation depending on whether convex hulls for point clouds exist or not
    if path_hull_1 is not None and path_hull_2 is not None:
        print("Loading hulls...")
        with open(path_hull_1, "rb") as file:
            hull_1 = pickle.load(file)
        with open(path_hull_2, "rb") as file:
            hull_2 = pickle.load(file)

        print("Computing distance 1 from hulls...")
        distances = []
        for i in tqdm(range(len(hull_2.vertices))):
            vertex = cropped_xyz_2[hull_2.vertices[i]]
            distance, _ = tree_1.query(vertex)
            distances.append(distance)
        
        distance1 = np.max(distances)

        print("Computing distance 2 from hulls...")
        distances = []
        for i in tqdm(range(len(hull_1.vertices))):
            vertex = cropped_xyz_1[hull_1.vertices[i]]
            distance, _ = tree_2.query(vertex)
            distances.append(distance)
        
        distance2 = np.max(distances)

        print("Hausdorff distance computation using hulls complete.")

        return max(distance1, distance2)
    
    else:
        print("Computing distance between point cloud 1 and tree 2...")
        distances = []
        for i in tqdm(range(len(cropped_xyz_1))):
            distance, _ = tree_2.query(cropped_xyz_1[i])
            distances.append(distance)
        
        distance1 = np.max(distances)
        print("Computing distance between point cloud 2 and tree 1...")
        distances = []
        for i in tqdm(range(len(cropped_xyz_2))):
            distance, _ = tree_1.query(cropped_xyz_2[i])
            distances.append(distance)
        
        distance2 = np.max(distances)

        print("Hausdorff distance computation without hulls complete.")

        return max(distance1, distance2)


if __name__ == "__main__":
    if len(sys.argv) not in [1, 3, 5]:
        print("Wrong number of inputs")
        sys.exit()
    if len(sys.argv) == 1:
        path_pc_1 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
        path_pc_2 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
        path_hull_1 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_real.pkl'
        path_hull_2 = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_real.pkl'
        #path_hull_1 = None
        #path_hull_2 = None
    elif len(sys.argv) == 3:
        path_pc_1 = sys.argv[1]
        path_pc_2 = sys.argv[2]
        path_hull_1 = None
        path_hull_2 = None
    else:
        path_pc_1 = sys.argv[1]
        path_pc_2 = sys.argv[2]
        path_hull_1 = sys.argv[3]
        path_hull_2 = sys.argv[4]
    
    d_haussd = hausdorff_distance_with_tree(path_pc_1=path_pc_1, path_pc_2=path_pc_2,
                    path_hull_1=path_hull_1, path_hull_2=path_hull_2)
    print(d_haussd)