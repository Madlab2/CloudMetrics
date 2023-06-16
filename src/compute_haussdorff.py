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

# def hausdorff_distance(path_pc_1, path_pc_2, path_hull_1, path_hull_2):
#     """
#     Calculates the Hausdorff distance between two point clouds.

#     Args:
#         path_pc_1 (str): Path to the first point cloud las file.
#         path_pc_2 (str): Path to the second point cloud las file.
#         path_hull_1 (str): Path to the file containing the convex hull of the first point cloud (precomputed).
#         path_hull_2 (str): Path to the file containing the convex hull of the second point cloud (precomputed).

#     Returns:
#         float: The Hausdorff distance between the two point clouds.
#     """
#     cropped_points_1, cropped_points_2 = helper.import_and_prepare_point_clouds(path_pc_1, path_pc_2)
    
#     # extract purely the coordinates of the points
#     cropped_xyz_1 = cropped_points_1[['x', 'y', 'z']]
#     cropped_xyz_2 = cropped_points_2[['x', 'y', 'z']]

#     # load hulls
#     with open(path_hull_1, "rb") as file:
#         hull_1 = pickle.load(file)

#     with open(path_hull_2, "rb") as file:
#         hull_2 = pickle.load(file)

#     # this takes a lot of time
#     print("Computing Distance 1...")
#     distance1 = None
#     with tqdm(total=len(hull_2.vertices), desc='Distance 1') as pbar:
#         for vertex_idx in hull_2.vertices:
#             distance = directed_hausdorff(cropped_xyz_1, cropped_xyz_2[vertex_idx])[0]
#             if distance1 is None or distance > distance1:
#                 distance1 = distance
#             pbar.update(1)

#     print("Computing Distance 2...")
#     distance2 = None
#     with tqdm(total=len(hull_1.vertices), desc='Distance 2') as pbar:
#         for vertex_idx in hull_1.vertices:
#             distance = directed_hausdorff(cropped_xyz_2, cropped_xyz_1[vertex_idx])[0]
#             if distance2 is None or distance > distance2:
#                 distance2 = distance
#             pbar.update(1)

#     print("Done.")

#     return max(distance1, distance2)

def laspy_to_np_array(laspy_points):
    x = laspy_points['x']
    y = laspy_points['y']
    z = laspy_points['z']
    return np.column_stack((x, y, z))

def build_kdtree(points):
    return cKDTree(points)

def hausdorff_distance_with_tree(path_pc_1, path_pc_2, path_hull_1, path_hull_2):
    """
    Calculates the Hausdorff distance between two point clouds using their convex hulls and spatial data structures.

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
    
    # convert to np.array
    cropped_xyz_1 = laspy_to_np_array(cropped_xyz_1)
    cropped_xyz_2 = laspy_to_np_array(cropped_xyz_2)

    # Downsample the point clouds for faster KD Tree construction (has only small effect on result)
    ratio = 1
    print("Downsampling first point cloud...")
    sampled_indices_1 = np.random.choice(len(cropped_xyz_1), size=len(cropped_xyz_1) // ratio, replace=False)
    sampled_points_1 = cropped_xyz_1[sampled_indices_1]
    
    print("Downsampling second point cloud...")
    sampled_indices_2 = np.random.choice(len(cropped_xyz_2), size=len(cropped_xyz_2) // ratio, replace=False)
    sampled_points_2 = cropped_xyz_2[sampled_indices_2]
       

    # load hulls
    with open(path_hull_1, "rb") as file:
        hull_1 = pickle.load(file)
    with open(path_hull_2, "rb") as file:
        hull_2 = pickle.load(file)

    print("Building kd-tree for the first point cloud...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tree = executor.submit(build_kdtree, sampled_points_1).result()
    
    print("Computing distance 1...")
    
    # Find nearest neighbors for each vertex in the second hull
    distances = []
    for i in tqdm(range(len(hull_2.vertices))):
        vertex = cropped_xyz_2[hull_2.vertices[i]]
        distance, _ = tree.query(vertex)
        distances.append(distance)
    
    # Compute Hausdorff distance
    distance1 = np.max(distances)
    
    print("Building kd-tree for the second point cloud...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tree = executor.submit(build_kdtree, sampled_points_2).result()

    print("Computing distance 2...")
    distances = []
    for i in tqdm(range(len(hull_1.vertices))):
        vertex = cropped_xyz_1[hull_1.vertices[i]]
        distance, _ = tree.query(vertex)
        distances.append(distance)
    
    distance2 = np.max(distances)

    print("Hausdorff distance computation complete.")

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
    
    d_haussd = hausdorff_distance_with_tree(path_pc_1=path_pc_1, path_pc_2=path_pc_2,
                        path_hull_1=path_hull_1, path_hull_2=path_hull_2)
    print(d_haussd)