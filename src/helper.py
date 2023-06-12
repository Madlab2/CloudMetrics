import numpy as np
import laspy
import matplotlib.pyplot as plt


def plot_bev(point_cloud, class_descriptions):
    """
    Plot a Bird's Eye View (BEV) of a point cloud with colorization based on semantic labels.
    Parameters:
        - point_cloud (dict): Dictionary containing the point cloud data, including 'x', 'y', and 'classification'.
        - class_descriptions (dict): Dictionary mapping class labels to their descriptions.
    Returns:
        None (displays the plot)
    """
    x = point_cloud['x']
    y = point_cloud['y']
    labels = point_cloud['classification']

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


def crop_point_cloud(point_cloud, bounding_box):
    # Extract x, y, z coordinates from point cloud
    x = point_cloud.x
    y = point_cloud.y
    z = point_cloud.z

    # bounding box coordinates
    min_x, min_y, min_z, max_x, max_y, max_z = bounding_box

    # Select the points within the bounding box
    mask = ((x >= min_x) & (x <= max_x) &
            (y >= min_y) & (y <= max_y) &
            (z >= min_z) & (z <= max_z))

    return point_cloud[mask]