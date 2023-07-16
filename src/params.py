#!/usr/bin/env python
import numpy as np

############################################
# Data Preparation
############################################

# OFFSETS

# Order of application:
# 1. mirror 
# 2. apply offset

# specific values for this project. Needs to be adjusted for other data
FLIP_REAL = False
X_OFFSET_REAL = -674000
Y_OFFSET_REAL = -5405000
Z_OFFSET_REAL = 0
# specific values for this project. Needs to be adjusted for other data
FLIP_SYNTH = True
X_OFFSET_SYNTH = -0.3
Y_OFFSET_SYNTH = -1.4
Z_OFFSET_SYNTH = -1.8

# GRID Filter (removes non-overlapping tiles)
GRID_FILTER_SIZE = 5 # meters

############################################
# Metric & Distances
############################################

# M3C2 PARAMS
EVERY_NTH = [5000, 2500, 2500, 1, 1, 10, 100] # Road, Ground, Wall, Roof, Door, Window, Building Installation
CYL_RADIUS = 0.21 # meters
NORMAL_RADII = [0.21, 0.42, 0.84] # meters
MAX_DISTANCE = 10 # meters

# for accepting/rejecting a claswise M3C2 result
MIN_POINTS = 20
NAN_THRESHOLD = 0.9
MIN_NUM_DISTANCES = 50

# CLOUD-TO-CLOUD
SPARSING_C2C = 10 # 1/10

# IoU - voxel-based calculation
IOU_VOXEL_SIZE = 0.5 # meters

# WEIGHTS
CLASS_NUM_TO_WEIGHT = {
    0: 0.1,     # Road
    1: 0.1,     # Ground
    2: 0.2,     # Wall Surface
    3: 0.15,    # Roof Surface
    4: 0.15,    # Doors
    5: 0.15,    # Windows
    6: 0.15,    # Building Installation
}
DISTANCE_WEIGHTS = np.array([0.6, 0.4]) # M3C2, C2C
METRIC_WEIGHTS = np.array([0.6, 0.2, 0.2]) # M3C2, C2C, MIoU

# bounded growth rate
SLOPE_FACTOR = -0.2