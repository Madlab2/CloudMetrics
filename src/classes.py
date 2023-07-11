#!/usr/bin/env python
CLASS_DESCRIPTIONS_REAL = {
    1: 'Road',
    2: 'Ground',
    3: 'Road Installations',
    6: 'Wall Surface',
    7: 'Roof Surface',
    8: 'Doors',
    9: 'Windows',
    10: 'Building Installations'
}

CLASS_DESCRIPTIONS_SYNTH = {
    #0: 'None',
    1: 'Road',
    4: 'Walls',
    #9: 'Vegetation',
    14: 'Car',
    25: 'Ground',
    0: 'Building Installations',    # was 29
    31: 'Window',
    33: 'Door',
    37: 'Roof',
}

CLASSES_FOR_M3C2_REAL = {
    1: 'Road',                      # 1
    2: 'Ground',                    # 2
    #3: 'Road Installations',
    6: 'Wall Surface',              # 3
    7: 'Roof Surface',              # 4
    8: 'Doors',                     # 5
    9: 'Windows',                   # 6
    10: 'Building Installations'    # 7
}

CLASSES_FOR_M3C2_SYNTH = {
    #0: 'None',
    1: 'Road',                      # 1
    25: 'Ground',                   # 2
    4: 'Walls',                     # 3
    #9: 'Vegetation',
    #14: 'Car',
    37: 'Roof',                     # 4
    33: 'Door',                     # 5
    31: 'Window',                   # 6
    0: 'Building Installations'     # 7 (was 29)
}