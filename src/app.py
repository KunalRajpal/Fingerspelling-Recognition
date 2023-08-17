import os
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import json
import mediapipe
import matplotlib
import matplotlib.pyplot as plt
import random

from skimage.transform import resize
from mediapipe.framework.formats import landmark_pb2
from tensorflow import keras
from keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc

# Load the data
# pd Shape function outputs a tuple
dataset_df = pd.read_csv("./data/input/train.csv")
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head())

# Fetch sequence_id, file_id, phrase from first row
sequence_id, file_id, phrase = dataset_df.iloc[0][["sequence_id", "file_id", "phrase"]]
print(f"sequence_id: {sequence_id}, file_id: {file_id}, phrase: {phrase}")


# Fetch data from parquet file
sample_sequence_df = pq.read_table(
    f"./data/input/train_landmarks/{str(file_id)}.parquet",
    filters=[
        [("sequence_id", "=", sequence_id)],
    ],
).to_pandas()
print("Full sequence dataset shape is {}".format(sample_sequence_df.shape))
