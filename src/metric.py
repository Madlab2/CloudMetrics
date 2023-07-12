#!/usr/bin/env python
import logging
import numpy as np
import py4dgeo, open3d
from tqdm import tqdm
import os, sys
import helper, classes

logging.basicConfig(level=logging.INFO)

output_file_path = "../results/metrics_results.txt"
OUTPUT = "Metrics Results\n"

EVERY_NTH = 1000
MIN_POINTS = 20
NAN_THRESHOLD = 0.9
MIN_NUM_DISTANCES = 50
SPARSING_C2C = 1

CLASS_NUM_TO_WEIGHT = {
    0: 0.1,     # Road
    1: 0.1,     # Ground
    2: 0.2,     # Wall Surface
    3: 0.15,    # Roof Surface
    4: 0.15,    # Doors
    5: 0.15,    # Windows
    6: 0.15,    # Building Installation
}

METRIC_WEIGHTS = np.array([0.66, 0.33]) # M3C2, C2C

def compute_metric(real_pc_path, synth_pc_path, c2c_distance=None):
    global OUTPUT

    logging.info('Reading & preparing data')
    real_points_all_classes, synth_points_all_classes = helper.import_and_prepare_point_clouds(real_pc_path, synth_pc_path, shift_real=True, flip_synth=True, crop=True)
    
    logging.info("Computing Class-Wise M3C2 Distances")
    class_wise_distances_results, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes = m3c2_class_wise(real_points_all_classes, synth_points_all_classes)
    
    classes_to_ignore = []
    OUTPUT += "\nM3C2 Medians for each class:\n"
    for class_number, median_distance, mean_distance, stdev in class_wise_distances_results:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        num_distances = len(class_wise_distances_all[class_number])
        nan_ratio = np.sum(np.isnan(class_wise_distances_all[class_number]))/num_distances
        OUTPUT += f"\n\tClass {class_name} ({class_number}):\n\t\tMedian distance = {median_distance},\n\t\tMean distance = {mean_distance},\n\t\tStandard Deviation = {stdev},\n\t\t#distances: {num_distances},\n\t\tNan-Ratio: {nan_ratio}\n"
        if np.isnan(median_distance) or nan_ratio > NAN_THRESHOLD or num_distances < MIN_NUM_DISTANCES:
            classes_to_ignore.append(class_number)
    
    OUTPUT += "\nM3C2 Ignored classes (not enough non-NaN distances):\n"
    for class_number in classes_to_ignore:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        OUTPUT += f"\tClass {class_name} ({class_number})\n"

    OUTPUT += "\nM3C2 Skipped classes (too few points):\n"
    for class_number in skipped_classes:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        OUTPUT += f"\tClass {class_name} ({class_number})\n"
        classes_to_ignore.append(class_number)
    
    # We need to remove weights for classes where distance is NaN and ignored classes. Renormalize weights (sum==1)
    weights = np.array([CLASS_NUM_TO_WEIGHT.get(class_number) for class_number in CLASS_NUM_TO_WEIGHT if class_number not in classes_to_ignore])
    weights = weights / np.sum(weights)

    mean_m3C2 = 0.0
    weight_idx = 0

    for class_number, median_distance, _, _ in class_wise_distances_results:
        if class_number not in classes_to_ignore:
            #logging.debug(f"\tAdding to Mean: Class Number {class_number}, Weight {weights[weight_idx]}, Median {median_distance}")
            mean_m3C2 += weights[weight_idx] * np.abs(median_distance)
            weight_idx += 1
    OUTPUT += f"\nWeightedMeanM3C2 = {mean_m3C2}\n"

    # calculate cloud to cloud distance
    logging.info("Computing Cloud-to-Cloud Distance")
    c2c_median_distance, c2c_mean_dist, c2c_stdev = cloud_to_cloud_distance(real_points_all_classes, synth_points_all_classes)
    OUTPUT += f"\nCloud2Cloud Results:\n\tMedian Distance = {c2c_median_distance} \n\tMean Distance = {c2c_mean_dist} \n\tStandard Deviation = {c2c_stdev}\n"
    
    metrics_vector = np.array([mean_m3C2, c2c_mean_dist])
    weight_vector = METRIC_WEIGHTS/np.sum(METRIC_WEIGHTS) # normalize weights
    distance_metric = (weight_vector.T).dot(metrics_vector)

    OUTPUT += f"\nFinal Cloud Comparison Metric = {distance_metric}"

    return distance_metric

def cloud_to_cloud_distance(real_points_all_classes, synth_points_all_classes):
    logging.info("\tC2C: Creating Open3D Point Clouds")
    real_points_np = laspy_to_np_array(real_points_all_classes, sparsing_factor=SPARSING_C2C)
    synth_points_np = laspy_to_np_array(synth_points_all_classes, sparsing_factor=SPARSING_C2C)

    real_cloud_o3d = open3d.geometry.PointCloud()
    real_cloud_o3d.points = open3d.utility.Vector3dVector(real_points_np)
    synth_cloud_o3d = open3d.geometry.PointCloud()
    synth_cloud_o3d.points = open3d.utility.Vector3dVector(synth_points_np)

    reference_pc = real_cloud_o3d
    target_pc = synth_cloud_o3d
    logging.info("\tC2C: Calculating Cloud-to-Cloud Distnace")
    distances = reference_pc.compute_point_cloud_distance(target_pc)
    median_distance = np.nanmedian(distances)
    mean_dist = np.nanmean(distances)
    stdev = np.nanstd(distances)
    return median_distance, mean_dist, stdev

def laspy_to_np_array(laspy_points, sparsing_factor=1):
    x = laspy_points['x'][::sparsing_factor]
    y = laspy_points['y'][::sparsing_factor]
    z = laspy_points['z'][::sparsing_factor]
    return np.column_stack((x, y, z))

def class_split_pc(points_all_classes, type=None):
    if type == 'real':
        labels = points_all_classes['classification']
        ordered_classes_dict = classes.CLASSES_FOR_M3C2_REAL
    elif type == 'synth':
        labels = points_all_classes.semantic_tags
        ordered_classes_dict = classes.CLASSES_FOR_M3C2_SYNTH
    else:
        return
    
    class_wise_points = []
    for class_number in ordered_classes_dict:
        class_indices = np.isin(labels, class_number)
        class_points = points_all_classes[class_indices]
        class_points_np = laspy_to_np_array(class_points)
        class_wise_points.append(class_points_np)
    
    return class_wise_points



def m3c2_class_wise(real_points_all_classes, synth_points_all_classes):    
    logging.info('\tM3C2 Splitting data')
    real_points_class_wise = class_split_pc(real_points_all_classes, type='real')
    synth_points_class_wise = class_split_pc(synth_points_all_classes, type='synth')

    # ensure number of point clouds/classes is the same for real and synth
    assert(len(real_points_class_wise) == len(synth_points_class_wise))
    
    class_wise_distances_all = []
    class_wise_distances_results = []
    class_wise_uncertainties_all = []
    skipped_classes = []

    logging.info('\tM3C2 Calculating distances...')
    for class_number in tqdm(range(len(real_points_class_wise)), bar_format='\t\t{l_bar}{bar}'):
        
        if len(real_points_class_wise[class_number]) < MIN_POINTS or len(synth_points_class_wise[class_number]) < MIN_POINTS:
            # not enough points for meaningful calculation
            logging.info("M3C2 Class {} has not enough points and is skipped".format(classes.CLASSES_FOR_M3C2_REAL[list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]]))
            skipped_classes.append(class_number)
        else:
            # m3c2 needs special epoch data type, timestamp is optional 
            epoch1 = py4dgeo.Epoch(real_points_class_wise[class_number])
            epoch2 = py4dgeo.Epoch(synth_points_class_wise[class_number])
            
            corepoints = epoch1.cloud[::EVERY_NTH]

            #TODO adjust params
            m3c2 = py4dgeo.M3C2(epochs=(epoch1, epoch2),
                corepoints=corepoints,
                cyl_radii=(2.0,),
                normal_radii=(0.5, 1.0, 2.0)
                )
            # run M3C2 calculation and suppress output
            py4dgeo_logger = logging.getLogger("py4dgeo")
            py4dgeo_logger.setLevel(logging.WARNING)
            distances, uncertainties = m3c2.run()
            py4dgeo_logger.setLevel(logging.INFO)

            distances = np.array(distances)
            uncertainties = np.array(uncertainties)
            # we have nan values, thus special median calculation
            median_distance = np.nanmedian(distances)
            mean_distance = np.nanmean(distances)
            stdev = np.nanstd(distances)
            #median_uncertainty = np.median(uncertainties["lodetection"])

            class_wise_distances_all.append(distances)
            class_wise_distances_results.append([class_number, median_distance, mean_distance, stdev])
            class_wise_uncertainties_all.append(uncertainties)
            

    return class_wise_distances_results, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes


if __name__ == "__main__":
    if len(sys.argv) not in [1, 3]:
        logging.info("Wrong number of inputs")
        sys.exit()
    if len(sys.argv) == 1:
        real_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/train/Train1 - labelled.las'
        synth_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Data Sets/Synthetic/Val_1 - Cloud.las'
    elif len(sys.argv) == 3:
        real_pc_path = sys.argv[1]
        synth_pc_path = sys.argv[2]
    
    mean_m3C2 = compute_metric(real_pc_path, synth_pc_path, c2c_distance=None)

    #global OUTPUT
    with open(output_file_path, "w") as file:
        file.write(OUTPUT)
    logging.info(f"Cloud Distance Metric: {mean_m3C2}")
    

    