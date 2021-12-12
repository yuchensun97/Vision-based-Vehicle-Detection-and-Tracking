"""
File and directory name
"""

DATA_DIR = "data"
VEHICLES_DATA_DIR = "vehicles"
NON_VEHICLES_DATA_DIR = "non-vehicles"
TEST_DATA_DIR = "test_images"
EXPORT_DIR = "export"
MODEL_DIR = "modmodels"

"""
Feature extraction
"""
ORIENTATION = 9
PIXEL_PER_CELL = 16
CELL_PER_BLOCK = 2
SPATIAL_SIZE = 16
HIST_BINS = 32

"""
Train
"""
MAX_ITER = 10000

"""
Infer
"""
SCALE = 1.8
WINDOW_SIZE = 64
CELL_PER_STEP = 2
PIXEL_PER_STEP = PIXEL_PER_CELL * CELL_PER_STEP
BBOX_THRESH = 50

"""
Plot
"""
BATCH_SIZE = 10
SAMPLE_SIZE = 10

"""
iou tracker
"""
t_min = 10