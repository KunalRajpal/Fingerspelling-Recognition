import os
import json
import pandas as pd

import os

# Determine the directory containing the data_loader.py script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path to the config.json file
config_file_path = os.path.join(script_dir, "..", "..", "config.json")


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


config = load_config(config_file_path)

BASE_DATA_PATH = config["base_data_path"]

# For CSV Files
TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, config["csv_files"]["train"])
SUPPLEMENTAL_METADATA_CSV_PATH = os.path.join(
    BASE_DATA_PATH, config["csv_files"]["supplemental_metadata"]
)

# For Parquet files
# (You can format these paths as needed with file_id or other variables in your script)
TRAIN_LANDMARK_PATTERN = os.path.join(
    BASE_DATA_PATH, config["parquet_patterns"]["train_landmarks"]
)
SUPPLEMENTAL_LANDMARK_PATTERN = os.path.join(
    BASE_DATA_PATH, config["parquet_patterns"]["supplemental_landmarks"]
)

# Load the data
# pd Shape function outputs a tuple
dataset_df = pd.read_csv(TRAIN_CSV_PATH)
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head())
