import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Modality Paths
# CORRECT line:
IMG_DIR = os.path.join(DATA_DIR, 'imaging')
IMG_CSV = os.path.join(DATA_DIR, 'imaging', 'Data_Entry_2017.csv') 
SENSOR_FILE = os.path.join(DATA_DIR, 'sensors', 'mitbih_test.csv')
EMR_FILE = os.path.join(DATA_DIR, 'emr', 'ADMISSIONS.csv')

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 4  # <--- CENTRAL CONTROL (Keep small for CPU)