import numpy as np
import laspy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pickle
from tqdm import tqdm


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

def test_script(path_pc1, path_pc2, classes):
    relevant_classes = list(class_descriptions.keys())

    real_pc = laspy.read(real_pc_path)
    synth_pc = laspy.read(synth_pc_path)

    # borders real
    header_real = real_pc.header
    min_borders_real = header_real.min
    max_borders_real = header_real.max
    
    # points real
    real_points = real_pc.points
    real_labels = real_points['classification']
    relevant_indices = np.isin(real_labels, relevant_classes)
    relevant_points_real = real_points[relevant_indices]

    # borders synth
    header_synth = synth_pc.header
    min_borders_synth = header_synth.min
    max_borders_synth = header_synth.max

    # points synth
    synth_points = synth_pc.points
    synth_labels = synth_points['classification']
    relevant_indices_synth = np.isin(synth_labels, relevant_classes)
    relevant_points_synth = synth_points[relevant_indices_synth]

    # bounding box & rectangular crop
    padding = 1.0
    min_xyz = np.maximum(min_borders_real, min_borders_synth) 
    max_xyz = np.minimum(max_borders_real, max_borders_synth)
    bounding_box = np.add(np.concatenate((min_xyz, max_xyz)), [-padding, -padding, -padding, padding, padding, padding])
    real_points_cropped = crop_points(relevant_points_real, bounding_box)
    synth_points_cropped = crop_points(relevant_points_synth, bounding_box)

    # extract purely the coordinates of the points
    real_xyz_cropped = real_points_cropped[['x', 'y', 'z']]
    synth_xyz_cropped = synth_points_cropped[['x', 'y', 'z']]

    convex_hull_path_real = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_real.pkl'
    convex_hull_path_synth = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/convex_hull_synth.pkl'

    num_partitions = 1000

    # Compute ConvexHull for real_xyz_cropped
    print("Computing ConvexHull for real_xyz_cropped...")
    convex_hull_real = None
    partition_size = len(real_xyz_cropped) // num_partitions
    for i in tqdm(range(num_partitions)):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        partition = np.column_stack((real_xyz_cropped[start_idx:end_idx]['x'],
                                 real_xyz_cropped[start_idx:end_idx]['y'],
                                 real_xyz_cropped[start_idx:end_idx]['z']))

        if convex_hull_real is None:
            convex_hull_real = ConvexHull(partition, incremental=True)
        else:
            convex_hull_real.add_points(partition, restart=False)

    # Close the convex hull computation
    convex_hull_real.close()
    
    with open(convex_hull_path_real, "wb") as file:
        pickle.dump(convex_hull_real, file)
    print(f"ConvexHull real data saved to: {convex_hull_path_real}")

    print("Computing ConvexHull for synth_xyz_cropped...")
    convex_hull_synth = None
    partition_size = len(synth_xyz_cropped) // num_partitions
    for i in tqdm(range(num_partitions)):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        partition = np.column_stack((synth_xyz_cropped[start_idx:end_idx]['x'],
                                 synth_xyz_cropped[start_idx:end_idx]['y'],
                                 synth_xyz_cropped[start_idx:end_idx]['z']))
        
        if convex_hull_synth is None:
            convex_hull_synth = ConvexHull(partition, incremental=True)
        else:
            convex_hull_synth.add_points(partition, restart=False)

    # Close the convex hull computation
    convex_hull_synth.close()

    with open(convex_hull_path_synth, "wb") as file:
        pickle.dump(convex_hull_synth, file)
    print(f"ConvexHull synth data saved to: {convex_hull_path_synth}")

    return hausdorff_distance(real_xyz_cropped, synth_xyz_cropped, convex_hull_real, convex_hull_synth)





if __name__ == "__main__":
    real_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
    synth_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/valid/validation_classified_merge.las'
    class_descriptions = {
        1: 'Road',
        2: 'Ground',
        #3: 'Road Installations',
        6: 'Wall Surface',
        7: 'Roof Surface',
        8: 'Doors',
        9: 'Windows',
        10: 'Building Installations'
    }

    test_script(path_pc1=real_pc_path, path_pc2=synth_pc_path, classes=class_descriptions)