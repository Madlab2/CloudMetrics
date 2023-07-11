#!/usr/bin/env python
import logging
import numpy as np
import py4dgeo
import pickle
from tqdm import tqdm
import contextlib
import os, sys
import helper, classes

logging.basicConfig(level=logging.INFO)

output_file_path = "../results/metrics_results.txt"  # Replace with the desired file path
OUTPUT = "Metrics Results\n"

EVERY_NTH = 10000
MIN_POINTS = 20
NAN_THRESHOLD = 0.9
MIN_NUM_DISTANCES = 50

CLASS_NUM_TO_WEIGHT = {
    0: 0.1,     # Road
    1: 0.1,     # Ground
    2: 0.2,     # Wall Surface
    3: 0.15,    # Roof Surface
    4: 0.15,    # Doors
    5: 0.15,    # Windows
    6: 0.15,    # Building Installation
}

def compute_metric(real_pc_path, synth_pc_path, c2c_distance=None):
    global OUTPUT

    logging.info('Reading & preparing data')
    real_points_all_classes, synth_points_all_classes = helper.import_and_prepare_point_clouds(real_pc_path, synth_pc_path, shift_real=True, flip_synth=True, crop=True)
    
    logging.info("Computing Class-Wise M3C2 Distances")
    class_wise_distances_medians, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes = m3c2_class_wise(real_points_all_classes, synth_points_all_classes)
    
    classes_to_ignore = []
    OUTPUT += "\nM3C2 Medians for each class:\n"
    for class_number, median_distance in class_wise_distances_medians:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        num_distances = len(class_wise_distances_all[class_number])
        nan_ratio = np.sum(np.isnan(class_wise_distances_all[class_number]))/num_distances
        OUTPUT += f"\tClass {class_name} ({class_number}):\tMedian distance = {median_distance},\t#distances: {num_distances},\tNan-Ratio: {nan_ratio}\n"
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

    for class_number, median_distance in class_wise_distances_medians:
        if class_number not in classes_to_ignore:
            #logging.debug(f"\tAdding to Mean: Class Number {class_number}, Weight {weights[weight_idx]}, Median {median_distance}")
            mean_m3C2 += weights[weight_idx] * np.abs(median_distance)
            weight_idx += 1
    OUTPUT += f"\nWeightedMeanM3C2 = {mean_m3C2}\n"
    return mean_m3C2

def laspy_to_np_array(laspy_points):
    x = laspy_points['x']
    y = laspy_points['y']
    z = laspy_points['z']
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
    class_wise_distances_medians = []
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
            #median_uncertainty = np.median(uncertainties["lodetection"])

            class_wise_distances_all.append(distances)
            class_wise_distances_medians.append([class_number, median_distance])
            class_wise_uncertainties_all.append(uncertainties)
            

    return class_wise_distances_medians, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes

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
    

    