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

def plot_bev_synth(points, class_list):
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
    labels = points.semantic_tags

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
    plt.legend(legend_colors, list(class_list.values()), loc='upper right', title='Classes')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Filtered Point Cloud (Bird\'s Eye View)')
    plt.axis('equal')
    plt.show()

def print_dims(pc):
    header = pc.header
    offsets = [header.x_offset, header.y_offset, header.z_offset]
    min_borders = header.min
    max_borders = header.max

    print("Point Cloud Min, Max, Offset:")
    print(min_borders)
    print(max_borders)
    print(offsets)
    return

def apply_offset(point_cloud, x_offset, y_offset, z_offset):

    # copy point cloud (do not change original)
    point_cloud_shifted = point_cloud
    # update header info (essential for not getting overflow errors)
    header_shifted = point_cloud_shifted.header
    new_offset = np.array([x_offset, y_offset, z_offset])
    old_offset = np.array(header_shifted.offset)
    global_shift = old_offset + new_offset
    header_shifted.offset = global_shift.tolist()
    header_shifted.min += new_offset
    header_shifted.max += new_offset
    point_cloud_shifted.header = header_shifted

    # Access the x, y, and z coordinates of the points
    points = np.vstack((point_cloud_shifted.x, point_cloud_shifted.y, point_cloud_shifted.z)).T

    # Apply the offset
    offset = np.array([x_offset, y_offset, z_offset])
    points += offset

    point_cloud_shifted.x = points[:, 0]
    point_cloud_shifted.y = points[:, 1]
    point_cloud_shifted.z = points[:, 2]

    return point_cloud_shifted

def apply_y_flip(point_cloud):
    
    # copy point cloud (do not change original)
    point_cloud_flipped = point_cloud
    # update header info
    header_flipped = point_cloud_flipped.header
    header_flipped.min[1] = -header_flipped.min[1]
    header_flipped.max[1] = -header_flipped.max[1]
    point_cloud_flipped.header = header_flipped
    y_points = np.array(point_cloud_flipped.points.y)
    # flip y
    point_cloud_flipped.points.y = -y_points
    return point_cloud_flipped

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