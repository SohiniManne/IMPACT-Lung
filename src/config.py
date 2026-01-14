import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Modality Paths
IMG_DIR = os.path.join(DATA_DIR, 'imaging')
# You need the NIH CSV file specifically
IMG_CSV = os.path.join(IMG_DIR, 'Data_Entry_2017.csv') 

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 8