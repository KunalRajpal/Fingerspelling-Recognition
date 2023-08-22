import os
import json
import pandas as pd
import os
import pyarrow.parquet as pq

# custom imports
from utils.logger import setup_logger

logger = setup_logger()

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

# Data input path
INPUT_PATH = os.path.join(BASE_DATA_PATH, "input")

# Preprocessed output path
PREPROCESSED_PATH = os.path.join(BASE_DATA_PATH, "preprocessed")
TF_FILE = os.path.join(PREPROCESSED_PATH, config["pre-processing"]["tf_file"])

# For CSV Files
TRAIN_CSV_PATH = os.path.join(INPUT_PATH, config["csv_files"]["train"])
SUPPLEMENTAL_METADATA_CSV_PATH = os.path.join(
    INPUT_PATH, config["csv_files"]["supplemental_metadata"]
)

# For Parquet files
# (You can format these paths as needed with file_id or other variables in your script)
TRAIN_LANDMARK_PATTERN = os.path.join(
    INPUT_PATH, config["parquet_patterns"]["train_landmarks"]
)
SUPPLEMENTAL_LANDMARK_PATTERN = os.path.join(
    INPUT_PATH, config["parquet_patterns"]["supplemental_landmarks"]
)

# Load the data
# pd Shape function outputs a tuple
dataset_df = pd.read_csv(TRAIN_CSV_PATH)
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head())

# Fetch sequence_id, file_id, phrase from first row
sequence_id, file_id, phrase = dataset_df.iloc[0][["sequence_id", "file_id", "phrase"]]
logger.info(f"sequence_id: {sequence_id}, file_id: {file_id}, phrase: {phrase}")

# Fetch data from parquet file
sample_parquet_file = TRAIN_LANDMARK_PATTERN.format(file_id=file_id)
sample_sequence_df = pq.read_table(
    sample_parquet_file,
    filters=[
        [("sequence_id", "=", sequence_id)],
    ],
).to_pandas()
logger.info("Full sequence dataset shape is {}".format(sample_sequence_df.shape))
logger.info("# frames:{}".format(sample_sequence_df.shape[0]))
logger.info("# columns:{}".format(sample_sequence_df.shape[1]))

# assertions to make sure everything is running smoothly
assert "sequence_id" in dataset_df.columns, "sequence_id not in csv columns"
assert "file_id" in dataset_df.columns, "file_id not in csv columns"
assert "phrase" in dataset_df.columns, "phrase not in csv columns"
