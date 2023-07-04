#!/usr/bin/env python
import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import classes, offsets


def plot_bev(points, type=None):
    """
    Plot a Bird's Eye View (BEV) of a point cloud with colorization based on semantic labels.
    Parameters:
        - points (dict): Dictionary containing the point cloud data points, including 'x', 'y', and 'classification'.
        - type (str): 'real' or 'synth' . Used for class desccriptions and semantic tag retrieval
    Returns:
        None (displays the plot)
    """
    x = points['x']
    y = points['y']
    if type == 'real':
        labels = points['classification']
    elif type == 'synth':
        labels = points.semantic_tags
    else: 
        # quit
        return

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

    if type == 'real':
        class_description_value_list = list(classes.CLASS_DESCRIPTIONS_REAL.values())
    else:
        class_description_value_list = list(classes.CLASS_DESCRIPTIONS_SYNTH.values())
    
    plt.legend(legend_colors, class_description_value_list, loc='upper right', title='Classes')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud (Bird\'s Eye View)')
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
    y_points = np.array(point_cloud_flipped.y)
    # flip y
    point_cloud_flipped.y = -y_points
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
    x = np.abs(points.x)
    y = np.abs(points.y)
    z = np.abs(points.z)

    # bounding box coordinates
    min_x, min_y, min_z, max_x, max_y, max_z = np.abs(bounding_box)

    # Select the points within the bounding box
    mask = ((x >= min_x) & (x <= max_x) &
            (y >= min_y) & (y <= max_y) &
            (z >= min_z) & (z <= max_z))

    return points[mask]


def import_and_prepare_point_clouds(path_pc_real, path_pc_synth, shift_real=True, flip_synth=True):
    """
    Imports and prepares two point clouds for comparison.

    Args:
        path_pc_real (str): Path to the real point cloud file.
        path_pc_synth (str): Path to the synthetic point cloud file.
        shift_real (bool): flag indicating whether to shift real point cloud by OFFSETS defined in offsets.py
        flipy_synth (bool): flag indicating whether to flip synthetic point cloud along y axis

    Returns:
        tuple: A tuple containing two arrays of cropped points from the input point clouds.
    """
    class_indices_real = list(classes.CLASS_DESCRIPTIONS_REAL.keys())
    class_indices_synth = list(classes.CLASS_DESCRIPTIONS_SYNTH.keys())
    pc_real = laspy.read(path_pc_real)
    pc_synth = laspy.read(path_pc_synth)

    # ensure that both pc are in same reference frame
    if shift_real == True:
        pc_real = apply_offset(pc_real, offsets.X_OFFSET_REAL, offsets.Y_OFFSET_REAL, offsets.Z_OFFSET_REAL)
    if flip_synth == True:
        pc_synth = apply_y_flip(pc_synth)
    
    # borders real pc
    header_real = pc_real.header
    min_borders_real = header_real.min
    max_borders_real = header_real.max
    
    # points real pc
    points_real = pc_real.points
    labels_real = points_real['classification']
    filter_indices_real = np.isin(labels_real, class_indices_real)
    filtered_points_real = points_real[filter_indices_real]

    # borders synth pc
    header_synth = pc_synth.header
    min_borders_synth = header_synth.min
    max_borders_synth = header_synth.max

    # points synth pc
    points_synth = pc_synth.points
    labels_synth = points_synth.semantic_tags
    filter_indices_synth = np.isin(labels_synth, class_indices_synth)
    filtered_points_synth = points_synth[filter_indices_synth]

    # bounding box & rectangular crop
    padding = 1.0
    min_xyz = np.maximum(min_borders_real, min_borders_synth) 
    max_xyz = np.minimum(max_borders_real, max_borders_synth)
    bounding_box = np.add(np.concatenate((min_xyz, max_xyz)), [-padding, -padding, -padding, padding, padding, padding])
    cropped_points_real = crop_points(filtered_points_real, bounding_box)
    cropped_points_synth = crop_points(filtered_points_synth, bounding_box)

    return cropped_points_real, cropped_points_synth