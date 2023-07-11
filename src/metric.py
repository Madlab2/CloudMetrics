#!/usr/bin/env python
import numpy as np
import py4dgeo
import pickle
from tqdm import tqdm
import os, sys
import helper, classes

EVERY_NTH = 10000
MIN_POINTS = 10


def compute_metric(dir_m3c2_real, dir_m3c2_synth, c2c_distance):
    return 0

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



def m3c2_class_wise(real_pc_path, synth_pc_path):
    print('Reading & preparing data')
    real_points_all_classes, synth_points_all_classes = helper.import_and_prepare_point_clouds(real_pc_path, synth_pc_path, shift_real=True, flip_synth=True, crop=True)
    print('Splitting data')
    real_points_class_wise = class_split_pc(real_points_all_classes, type='real')
    synth_points_class_wise = class_split_pc(synth_points_all_classes, type='synth')

    # ensure number of point clouds/classes is the same for real and synth
    assert(len(real_points_class_wise) == len(synth_points_class_wise))
    
    class_wise_distances_all = []
    class_wise_distances_medians = []
    class_wise_uncertainties_all = []
    skipped_classes = []

    print('Calculating distances...')
    for class_number in tqdm(range(len(real_points_class_wise))):
        
        if len(real_points_class_wise[class_number]) < MIN_POINTS or len(synth_points_class_wise[class_number]) < MIN_POINTS:
            # not enough points for meaningful calculation
            print("Class {} has not enough points and is skipped".format(classes.CLASSES_FOR_M3C2_REAL[list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]]))
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
            distances, uncertainties = m3c2.run()

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
        print("Wrong number of inputs")
        sys.exit()
    if len(sys.argv) == 1:
        real_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Labelling/Label-Datasets/train/Train1 - labelled.las'
        synth_pc_path = '/home/Meins/Uni/TUM/SS23/Data Lab/Data Sets/Synthetic/Val_1 - Cloud.las'
    elif len(sys.argv) == 3:
        real_pc_path = sys.argv[1]
        synth_pc_path = sys.argv[2]
    
    class_wise_distances_medians, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes = m3c2_class_wise(real_pc_path, synth_pc_path)
    
    print("Medians for each class:")
    for class_number, median_distance in class_wise_distances_medians:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        num_distances = len(class_wise_distances_all[class_number])
        nan_ratio = np.sum(np.isnan(class_wise_distances_all[class_number]))/num_distances
        print(f"Class {class_name} ({class_number}):\tMedian distance = {median_distance},\t#distances: {num_distances},\tNan-Ratio: {nan_ratio}")
    print("Skipped classes:")
    for class_number in skipped_classes:
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]
        print(f"Class {class_name} ({class_number})")

    