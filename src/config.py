import os

# ---------------------------------------------------
# PATH CONFIGURATIONS
# ---------------------------------------------------
# Get the project root directory (assuming this file is in src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Classification Data Paths
# Ensure your folder structure matches: data/classification/train/bird, etc.
TRAIN_DIR = os.path.join(DATA_DIR, 'classification', 'train')
VAL_DIR = os.path.join(DATA_DIR, 'classification', 'val')
TEST_DIR = os.path.join(DATA_DIR, 'classification', 'test')

# Object Detection Data Paths (YOLO)
# This points to the config file required by YOLOv8
YOLO_YAML_PATH = os.path.join(DATA_DIR, 'object_detection', 'data.yaml')

# ---------------------------------------------------
# MODEL HYPERPARAMETERS
# ---------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Binary Classification: 0=Bird, 1=Drone (depending on folder alphabetical order)
# Usually, Keras assigns 0 to the first folder alphabetically.
CLASSES = ['Bird', 'Drone']