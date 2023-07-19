#!/usr/bin/env python
import logging
import numpy as np
import py4dgeo, open3d
from tqdm import tqdm
import os, sys
import helper, classes, params

logging.basicConfig(level=logging.INFO)

output_file_path = "../results/metrics_results.txt"
OUTPUT = "Metrics Results\n"



def iou_eval_function(mean_iou):
    #return (1 - mean_iou) * np.exp(1-2*mean_iou)
    epsilon = 0.00001
    return (1/(mean_iou + epsilon))

def metric_eval_function(mean_m3C2, c2c_mean_dist, m_iou_factor):
    # returns something between 0 and 1
    metrics_vec = np.abs(np.array([mean_m3C2, c2c_mean_dist, m_iou_factor]))
    weight_vec = params.METRIC_WEIGHTS.T
    scalar_prod = weight_vec.dot(metrics_vec)
    metric = 1 - np.exp(params.SLOPE_FACTOR * scalar_prod)
    return metric

def compute_metric(real_pc_path, synth_pc_path):
    global OUTPUT

    logging.info('Reading & preparing data')
    real_points_all_classes, synth_points_all_classes = helper.import_and_prepare_point_clouds(real_pc_path, synth_pc_path, crop=True)
    
    logging.info('Splitting data')
    real_points_class_wise = class_split_pc(real_points_all_classes, type='real')
    synth_points_class_wise = class_split_pc(synth_points_all_classes, type='synth')

    logging.info("Computing Class-Wise M3C2 Distances")
    class_wise_distances_results, class_wise_distances_all, class_wise_uncertainties_all, skipped_classes = m3c2_class_wise(real_points_class_wise, synth_points_class_wise)
    logging.info(f"Skipped Classes: {skipped_classes}")
    
    classes_to_ignore = []
    OUTPUT += "\nM3C2 Medians for each class:\n"
    idx = 0
    for class_number, median_distance, mean_distance, stdev in class_wise_distances_results:
        
        class_idx = list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_idx]

        num_distances = len(class_wise_distances_all[idx])
        nan_ratio = np.sum(np.isnan(class_wise_distances_all[idx]))/num_distances
        OUTPUT += f"\n\tClass {class_name} ({class_number}):\n\t\tMedian distance = {median_distance},\n\t\tMean distance = {mean_distance},\n\t\tStandard Deviation = {stdev},\n\t\t#distances: {num_distances},\n\t\tNan-Ratio: {nan_ratio}\n"
        if np.isnan(median_distance) or nan_ratio > params.NAN_THRESHOLD or num_distances < params.MIN_NUM_DISTANCES:
            classes_to_ignore.append(class_number)
        idx += 1
    
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
    weights = np.array([params.CLASS_NUM_TO_WEIGHT.get(class_number) for class_number in params.CLASS_NUM_TO_WEIGHT if class_number not in classes_to_ignore])
    weights = weights / np.sum(weights)

    mean_m3C2 = 0.0
    weight_idx = 0
    for class_number, median_distance, _, _ in class_wise_distances_results:
        if class_number not in classes_to_ignore:
            #logging.debug(f"\tAdding to Mean: Class Number {class_number}, Weight {weights[weight_idx]}, Median {median_distance}")
            mean_m3C2 += weights[weight_idx] * np.abs(median_distance)
            weight_idx += 1
    OUTPUT += f"\nWeightedMeanM3C2 = {mean_m3C2}\n"

    # calculate cloud to cloud distances
    logging.info("Computing Cloud-to-Cloud Distance forward")
    c2c_median_distance_fwd, c2c_mean_dist_fwd, c2c_stdev_fwd = cloud_to_cloud_distance(real_points_all_classes, synth_points_all_classes)
    OUTPUT += f"\nCloud2Cloud Results Forward:\n\tMedian Distance = {c2c_median_distance_fwd} \n\tMean Distance = {c2c_mean_dist_fwd} \n\tStandard Deviation = {c2c_stdev_fwd}\n"
    logging.info("Computing Cloud-to-Cloud Distance backward")
    c2c_median_distance_bkwd, c2c_mean_dist_bkwd, c2c_stdev_bkwd = cloud_to_cloud_distance(synth_points_all_classes, real_points_all_classes)
    OUTPUT += f"\nCloud2Cloud Results Backward:\n\tMedian Distance = {c2c_median_distance_bkwd} \n\tMean Distance = {c2c_mean_dist_bkwd} \n\tStandard Deviation = {c2c_stdev_bkwd}\n"
    
    c2c_mean_dist = np.mean([np.abs(c2c_mean_dist_fwd), np.abs(c2c_mean_dist_bkwd)])
    OUTPUT += f"Cloud2Cloud mean distance = {c2c_mean_dist}\n"
    
    # Interesection over Union
    logging.info("Computing MeanIoU")
    weighted_mean_iou, _ = class_wisevoxel_iou(real_points_class_wise, synth_points_class_wise)
    m_iou_factor = iou_eval_function(weighted_mean_iou)
    OUTPUT += f"\nWeighted MeanIoU = {round(100 * weighted_mean_iou, 4)} %\nMeanIoU Factor = {m_iou_factor}\n"

    logging.info(f"Calculating Metric")
    metric = metric_eval_function(mean_m3C2, c2c_mean_dist, m_iou_factor)

    logging.info(f"Calculating Distance")
    distance_vector = np.array([mean_m3C2, c2c_mean_dist])
    distance_weights = params.DISTANCE_WEIGHTS.T
    distance = distance_weights.dot(distance_vector)

    OUTPUT += "\n-----------------------------\nSummary:"
    OUTPUT += f"\nFinal Cloud Comparison Metric = {round(metric, 4)}"
    OUTPUT += f"\nWeighed Cloud Distance (M3C2 and C2C) = {round(distance, 4)}\n\n"
    OUTPUT += f"WeightedMeanM3C2 = {round(mean_m3C2, 4)}\n"
    OUTPUT += f"Cloud2Cloud mean distance = {round(c2c_mean_dist, 4)}\n"
    OUTPUT += f"Weighted MeanIoU = {round(100 * weighted_mean_iou, 4)}%"

    return metric, distance

def m3c2_class_wise(real_points_class_wise, synth_points_class_wise):    
    
    # ensure number of point clouds/classes is the same for real and synth
    assert(len(real_points_class_wise) == len(synth_points_class_wise))
    assert(len(real_points_class_wise) == len(params.EVERY_NTH))

    class_wise_distances_all = []
    class_wise_distances_results = []
    class_wise_uncertainties_all = []
    skipped_classes = []

    logging.info('\tM3C2: Calculating distances...')
    for class_number in tqdm(range(len(real_points_class_wise)), bar_format='\t\t{l_bar}{bar}'):
        
        if len(real_points_class_wise[class_number]) < params.MIN_POINTS or len(synth_points_class_wise[class_number]) < params.MIN_POINTS:
            # not enough points for meaningful calculation
            logging.info("M3C2: Class {} has not enough points and is skipped".format(classes.CLASSES_FOR_M3C2_REAL[list(classes.CLASSES_FOR_M3C2_REAL.keys())[class_number]]))
            skipped_classes.append(class_number)
        else:
            # m3c2 needs special epoch data type, timestamp is optional 
            epoch1 = py4dgeo.Epoch(real_points_class_wise[class_number])
            epoch2 = py4dgeo.Epoch(synth_points_class_wise[class_number])
            
            corepoints = epoch1.cloud[::params.EVERY_NTH[class_number]]

            m3c2 = py4dgeo.M3C2(epochs=(epoch1, epoch2),
                corepoints=corepoints,
                cyl_radii=(params.CYL_RADIUS,),
                normal_radii=params.NORMAL_RADII,
                max_distance=params.MAX_DISTANCE
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

def cloud_to_cloud_distance(real_points_all_classes, synth_points_all_classes):
    logging.info("\tC2C: Creating Open3D Point Clouds")
    real_points_np = laspy_to_np_array(real_points_all_classes, sparsing_factor=params.SPARSING_C2C)
    synth_points_np = laspy_to_np_array(synth_points_all_classes, sparsing_factor=params.SPARSING_C2C)

    real_cloud_o3d = open3d.geometry.PointCloud()
    real_cloud_o3d.points = open3d.utility.Vector3dVector(real_points_np)
    synth_cloud_o3d = open3d.geometry.PointCloud()
    synth_cloud_o3d.points = open3d.utility.Vector3dVector(synth_points_np)

    reference_pc = real_cloud_o3d
    target_pc = synth_cloud_o3d
    logging.info("\tC2C: Calculating Cloud-to-Cloud Distance")
    distances = reference_pc.compute_point_cloud_distance(target_pc)
    median_distance = np.nanmedian(distances)
    mean_dist = np.nanmean(distances)
    stdev = np.nanstd(distances)
    return median_distance, mean_dist, stdev

def class_wisevoxel_iou(real_points_class_wise, synth_points_class_wise):
    global OUTPUT
    idx = 0
    class_wise_iou = []
    
    logging.info("\tIoU: Calculating for each Class")
    OUTPUT += "\nIntersection over Union (IoU) for each class:"

    for class_points_real, class_points_synth in tqdm(zip(real_points_class_wise, synth_points_class_wise), bar_format='\t\t{l_bar}{bar}'):
     
        max_xyz = np.maximum(np.max(class_points_real, axis=0), np.max(class_points_synth, axis=0))
        min_xyz = np.minimum(np.min(class_points_real, axis=0), np.min(class_points_synth, axis=0))

        voxel_grid_dim_xyz = np.abs(max_xyz - min_xyz)
        
        grid_x = np.arange(0, voxel_grid_dim_xyz[0] + params.IOU_VOXEL_SIZE, params.IOU_VOXEL_SIZE) + min_xyz[0]
        grid_y = np.arange(0, voxel_grid_dim_xyz[1] + params.IOU_VOXEL_SIZE, params.IOU_VOXEL_SIZE) + min_xyz[1]
        grid_z = np.arange(0, voxel_grid_dim_xyz[2] + params.IOU_VOXEL_SIZE, params.IOU_VOXEL_SIZE) + min_xyz[2]
        
        hist_real, _ = np.histogramdd(class_points_real, bins=(grid_x, grid_y, grid_z))
        hist_synth, _ = np.histogramdd(class_points_synth, bins=(grid_x, grid_y, grid_z))
        
        intersection = np.sum(np.logical_and(hist_real > 0, hist_synth > 0))
        union = np.sum(np.logical_or(hist_real > 0, hist_synth > 0))
        iou_per_voxel = intersection / union
        class_wise_iou.append(iou_per_voxel)
        
        class_number = list(classes.CLASSES_FOR_M3C2_REAL.keys())[idx]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_number]
        OUTPUT += f"\n\tClass {class_name} ({class_number}):\t IoU = {round(100 * iou_per_voxel, 2)} %"
        idx += 1
    
    weights = np.array(list(params.CLASS_NUM_TO_WEIGHT.values())).T
    weighted_mean_iou = weights.dot(class_wise_iou)

    return weighted_mean_iou, class_wise_iou

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



if __name__ == "__main__":
    if len(sys.argv) not in [1, 3]:
        logging.info("Wrong number of inputs")
        sys.exit()
    if len(sys.argv) == 1:
        real_pc_path = params.DEFAULT_REAL_PC_PATH
        synth_pc_path = params.DEFAULT_SYNTH_PC_PATH
    elif len(sys.argv) == 3:
        real_pc_path = sys.argv[1]
        synth_pc_path = sys.argv[2]
    
    metric, distance = compute_metric(real_pc_path, synth_pc_path)
    
    with open(output_file_path, "w") as file:
        file.write(OUTPUT)
    logging.info(f"Cloud Distance: {round(distance, 4)} m")
    logging.info(f"Cloud Metric: {round(metric, 4)}")
    

    