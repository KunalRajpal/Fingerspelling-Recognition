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
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from matplotlib import animation, rc
from IPython.display import display, HTML

# Load the data
# pd Shape function outputs a tuple
# use absolute path here
dataset_df = pd.read_csv(
    "/Users/kunalrajpal/Documents/Tech/Fingerspelling-recognition/data/input/train.csv"
)
print("Full train dataset shape is {}".format(dataset_df.shape))
print(dataset_df.head())

# Fetch sequence_id, file_id, phrase from first row
sequence_id, file_id, phrase = dataset_df.iloc[0][["sequence_id", "file_id", "phrase"]]
print(f"sequence_id: {sequence_id}, file_id: {file_id}, phrase: {phrase}")


# Fetch data from parquet file
# use absolute path here
sample_sequence_df = pq.read_table(
    f"/Users/kunalrajpal/Documents/Tech/Fingerspelling-recognition/data/input/train_landmarks/{str(file_id)}.parquet",
    filters=[
        [("sequence_id", "=", sequence_id)],
    ],
).to_pandas()
print("Full sequence dataset shape is {}".format(sample_sequence_df.shape))
print("# frames:{}".format(sample_sequence_df.shape[0]))
print("# columns:{}".format(sample_sequence_df.shape[1]))

# assertions to make sure everything is running smoothly
assert "sequence_id" in dataset_df.columns, "sequence_id not in csv columns"
assert "file_id" in dataset_df.columns, "file_id not in csv columns"
assert "phrase" in dataset_df.columns, "phrase not in csv columns"


# Function create animation from images.

matplotlib.rcParams["animation.embed_limit"] = 2**128
matplotlib.rcParams["savefig.pad_inches"] = 0
rc("animation", html="jshtml")


def create_animation(images):
    fig = plt.figure(figsize=(6, 9))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(images[0], cmap="gray")
    plt.close(fig)

    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=1000 / 10
    )


# Extract the landmark data and convert it to an image using mediapipe library.
# This function extracts the data for both hands.

mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles


def get_hands(seq_df):
    images = []
    all_hand_landmarks = []
    for seq_idx in range(len(seq_df)):
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_right_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_right_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_right_hand.*").values

        right_hand_image = np.zeros((600, 600, 3))

        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()

        for x, y, z in zip(x_hand, y_hand, z_hand):
            right_hand_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
            right_hand_image,
            right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        )

        x_hand = seq_df.iloc[seq_idx].filter(regex="x_left_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_left_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_left_hand.*").values

        left_hand_image = np.zeros((600, 600, 3))

        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_hand, y_hand, z_hand):
            left_hand_landmarks.landmark.add(x=x, y=y, z=z)

        mp_drawing.draw_landmarks(
            left_hand_image,
            left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        )

        images.append(
            [right_hand_image.astype(np.uint8), left_hand_image.astype(np.uint8)]
        )
        all_hand_landmarks.append([right_hand_landmarks, left_hand_landmarks])
    return images, all_hand_landmarks


# Get the images created using mediapipe apis
hand_images, hand_landmarks = get_hands(sample_sequence_df)
# Fetch and show the data for right hand
anim = create_animation(np.array(hand_images)[:, 0])
display(HTML(anim.to_jshtml()))  # This will display the animation
