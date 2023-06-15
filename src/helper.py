#!/usr/bin/env python
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import classes


def plot_bev(points):
    """
    Plot a Bird's Eye View (BEV) of a point cloud with colorization based on semantic labels.
    Parameters:
        - points (dict): Dictionary containing the point cloud data points, including 'x', 'y', and 'classification'.
        - class_descriptions (dict): Dictionary mapping class labels to their descriptions.
    Returns:
        None (displays the plot)
    """
    x = points['x']
    y = points['y']
    labels = points['classification']

    # Plot every 100th point with colorization
    stride = 100
    x_sampled = x[::stride]
    y_sampled = y[::stride]
    labels_sampled = labels[::stride]

    # Create a scatter plot in bird's eye view with colorization
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x_sampled, y_sampled, c=labels_sampled, cmap='Set1', s=0.1)

    # Add a legend for the semantic labels
    legend_colors = scatter.legend_elements(prop='colors')[0]
    plt.legend(legend_colors, list(classes.CLASS_DESCRIPTIONS.values()), loc='upper right', title='Classes')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Filtered Point Cloud (Bird\'s Eye View)')
    plt.axis('equal')
    plt.show()


def crop_points(points, bounding_box):
    """
    Crop a point cloud based on a given bounding box.

    Parameters:
        points (LasPointRecord): The input point cloud data points.
        bounding_box (list): The bounding box coordinates in the order [min_x, min_y, min_z, max_x, max_y, max_z].

    Returns:
        LasPointRecord: The cropped point cloud data points containing points within the specified bounding box.
    
    Note:
        - The function preserves all the data in the `points` object, including x, y, z coordinates, intensity,
        classification, and any other available attributes, for the points that are within the specified bounding box.
        - Header Information of the point cloud is not updated (e.g. min and max values)
    """
    # Extract x, y, z coordinates from points
    x = points.x
    y = points.y
    z = points.z

    # bounding box coordinates
    min_x, min_y, min_z, max_x, max_y, max_z = bounding_box

    # Select the points within the bounding box
    mask = ((x >= min_x) & (x <= max_x) &
            (y >= min_y) & (y <= max_y) &
            (z >= min_z) & (z <= max_z))

    return points[mask]

def hausdorff_distance(point_set1, point_set2, hull1, hull2):
    """
    Compute the Hausdorff distance between two point sets.
    
    The Hausdorff distance is a measure of dissimilarity between two sets of points in a metric space. It represents the
    maximum distance between a point in one set and its closest point in the other set.
    
    Parameters:
        point_set1 (ndarray): Array of shape (N, 3) representing the first set of points.
        point_set2 (ndarray): Array of shape (M, 3) representing the second set of points.
        hull1 (scipy.spatial.ConvexHull): Pre-computed ConvexHull object for point_set1.
        hull2 (scipy.spatial.ConvexHull): Pre-computed ConvexHull object for point_set2.
    
    Returns:
        float: The Hausdorff distance between the two point sets.
    """
    distance1 = directed_hausdorff(point_set1, point_set2[hull2.vertices])[0]
    distance2 = directed_hausdorff(point_set2, point_set1[hull1.vertices])[0]

    return max(distance1, distance2)


def import_and_prepare_point_clouds(path_pc_1, path_pc_2):
    """
    Imports and prepares two point clouds for comparison.

    Args:
        path_pc_1 (str): Path to the first point cloud file.
        path_pc_2 (str): Path to the second point cloud file.

    Returns:
        tuple: A tuple containing two arrays of cropped points from the input point clouds.
    """
    class_indices = list(classes.CLASS_DESCRIPTIONS.keys())

    pc_1 = laspy.read(path_pc_1)
    pc_2 = laspy.read(path_pc_2)

    # borders pc 1
    header_1 = pc_1.header
    min_borders_1 = header_1.min
    max_borders_1 = header_1.max
    
    # points pc 1
    points_1 = pc_1.points
    labels_1 = points_1['classification']
    filter_indices_1 = np.isin(labels_1, class_indices)
    filtered_points_1 = points_1[filter_indices_1]

    # borders pc 2
    header_2 = pc_2.header
    min_borders_2 = header_2.min
    max_borders_2 = header_2.max

    # points pc 2
    points_2 = pc_2.points
    labels_2 = points_2['classification']
    filter_indices_2 = np.isin(labels_2, class_indices)
    filtered_points_2 = points_2[filter_indices_2]

    # bounding box & rectangular crop
    padding = 1.0
    min_xyz = np.maximum(min_borders_1, min_borders_2) 
    max_xyz = np.minimum(max_borders_1, max_borders_2)
    bounding_box = np.add(np.concatenate((min_xyz, max_xyz)), [-padding, -padding, -padding, padding, padding, padding])
    cropped_points_1 = crop_points(filtered_points_1, bounding_box)
    cropped_points_2 = crop_points(filtered_points_2, bounding_box)

    return cropped_points_1, cropped_points_2