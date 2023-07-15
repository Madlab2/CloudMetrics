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

def grid_filter(points1, points2, min_xyz, max_xyz, grid_size=5):
    points1_2d = np.column_stack((points1.x, points1.y))
    points2_2d = np.column_stack((points2.x, points2.y))

    # Calculate the positive grid dimensions for X and Y
    grid_dim_x = np.abs(max_xyz[0] - min_xyz[0])
    grid_dim_y = np.abs(max_xyz[1] - min_xyz[1])

    for i in range(len(max_xyz)):
        if min_xyz[i] > max_xyz[i]:
            temp = min_xyz[i]
            min_xyz[i] = max_xyz[i]
            max_xyz[i] = temp

    # Create the grid based on the positive grid dimensions

    grid_x = np.arange(0, grid_dim_x + grid_size, grid_size) + min_xyz[0]
    grid_y = np.arange(0, grid_dim_y + grid_size, grid_size) + min_xyz[1]
   
    # Ensure that both point clouds have non-empty data
    if len(points1_2d) == 0 or len(points2_2d) == 0:
        return np.array([]), np.array([])

    # Create a 2D histogram for both point clouds
    hist1, x_edges, y_edges = np.histogram2d(points1_2d[:, 0], points1_2d[:, 1], bins=(grid_x, grid_y))
    hist2, _, _ = np.histogram2d(points2_2d[:, 0], points2_2d[:, 1], bins=(x_edges, y_edges))

    # Find the indices of grid cells containing points from both point clouds
    overlapping_grid_indices = (hist1 > 0) & (hist2 > 0)

    # Filter points from both point clouds based on the overlapping grid cells
    grid_indices_x = np.clip(np.digitize(points1_2d[:, 0], grid_x) - 1, 0, len(x_edges) - 2)
    grid_indices_y = np.clip(np.digitize(points1_2d[:, 1], grid_y) - 1, 0, len(y_edges) - 2)
    filtered_points1 = points1[overlapping_grid_indices[grid_indices_x, grid_indices_y]]

    grid_indices_x = np.clip(np.digitize(points2_2d[:, 0], grid_x) - 1, 0, len(x_edges) - 2)
    grid_indices_y = np.clip(np.digitize(points2_2d[:, 1], grid_y) - 1, 0, len(y_edges) - 2)
    filtered_points2 = points2[overlapping_grid_indices[grid_indices_x, grid_indices_y]]

    return filtered_points1, filtered_points2

def import_and_prepare_point_clouds(path_pc_real, path_pc_synth, shift_real=True, flip_synth=True, crop=False):
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
    

    # filter classes
    points_real = pc_real.points
    labels_real = points_real['classification']
    filter_indices_real = np.isin(labels_real, class_indices_real)
    filtered_points_real = points_real[filter_indices_real]
    
    points_synth = pc_synth.points
    labels_synth = points_synth.semantic_tags
    filter_indices_synth = np.isin(labels_synth, class_indices_synth)
    filtered_points_synth = points_synth[filter_indices_synth]

    if crop == False:
        return filtered_points_real, filtered_points_synth
    
    # continue with cropping
    header_real = pc_real.header
    min_borders_real = header_real.min
    max_borders_real = header_real.max
    
    header_synth = pc_synth.header
    min_borders_synth = header_synth.min
    max_borders_synth = header_synth.max

    # bounding box & rectangular crop
    padding = 1.0
    min_xyz = np.maximum(min_borders_real, min_borders_synth) 
    max_xyz = np.minimum(max_borders_real, max_borders_synth)

    cropped_points_real, cropped_points_synth = grid_filter(filtered_points_real, filtered_points_synth, min_xyz, max_xyz, grid_size=5)

    return cropped_points_real, cropped_points_synth