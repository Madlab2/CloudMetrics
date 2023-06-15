import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plot_bev(points, class_descriptions):
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
    plt.legend(legend_colors, list(class_descriptions.values()), loc='upper right', title='Classes')

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
        The function preserves all the data in the `points` object, including x, y, z coordinates, intensity,
        classification, and any other available attributes, for the points that are within the specified bounding box.
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

def hausdorff_distance(point_set1, point_set2, hull1=None, hull2=None):
    """
    Compute the Hausdorff distance between two point sets.
    
    The Hausdorff distance is a measure of dissimilarity between two sets of points in a metric space. It represents the
    maximum distance between a point in one set and its closest point in the other set.
    
    Parameters:
        point_set1 (ndarray): Array of shape (N, 3) representing the first set of points.
        point_set2 (ndarray): Array of shape (M, 3) representing the second set of points.
        hull1 (scipy.spatial.ConvexHull, optional): Pre-computed ConvexHull object for point_set1. If not provided, it
            will be computed internally.
        hull2 (scipy.spatial.ConvexHull, optional): Pre-computed ConvexHull object for point_set2. If not provided, it
            will be computed internally.
    
    Returns:
        float: The Hausdorff distance between the two point sets.
    """
    if hull1 is None:
        hull1 = ConvexHull(point_set1)
    if hull2 is None:
        hull2 = ConvexHull(point_set2)
    
    closest_points_set1 = point_set2[hull2.closest_point(point_set1)]
    closest_points_set2 = point_set1[hull1.closest_point(point_set2)]
    
    distances1 = np.linalg.norm(point_set1 - closest_points_set1, axis=1)
    distances2 = np.linalg.norm(point_set2 - closest_points_set2, axis=1)
    
    max_distance = np.max(np.concatenate((distances1, distances2)))
    
    return max_distance


