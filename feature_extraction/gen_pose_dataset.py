import cv2
import numpy as np
import pickle
import sys
import os

import pandas as pd
import time
from numpy import random

sys.path.append("../")

from tracker import Tracker
from pose_estimation import PoseEstimator

from config import *


def print_progress(cur_idx, omit, L, start_time):
    time_left = int((time.time() - start_time) / (cur_idx - omit) * (L - cur_idx + omit))

    hours = time_left // 3600

    minutes = time_left % 3600 // 60

    seconds = time_left % 60

    print("\rProgress: %.2f" % ((cur_idx - omit) * 100 / L) + "% "
          + str(hours) + " hours "
          + str(minutes) + " minutes "
          + str(seconds) + " seconds left",
          end=" ")


def split_and_save(samples, labels, splits):
    assert len(samples) == len(labels) == len(splits), "Dimensions mismatch"
    X = np.array(samples)
    y = np.array(LABEL_ENCODER[label] for label in labels).astype(np.int8)
    train_idxs = [i for i in range(len(y)) if splits[i] == "train"]
    val_idxs = [i for i in range(len(y)) if splits[i] == "val"]
    test_idxs = [i for i in range(len(y)) if splits[i] == "test"]

    random.shuffle(train_idxs)
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)

    X_train = X[train_idxs]
    y_train = y[train_idxs]
    X_val = X[val_idxs]
    y_val = y[val_idxs]
    X_test = X[test_idxs]
    y_test = y[test_idxs]

    np.save("../vars/X_train", X_train)
    np.save("../vars/X_val", X_val)
    np.save("../vars/X_test", X_test)

    np.save("../vars/y_train", y_train)
    np.save("../vars/y_val", y_val)
    np.save("../vars/y_test", y_test)


if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join(DATASET_DIR, "annotation.csv"))
    start_time = time.time()
    omit = 0

    samples = []
    labels = []
    splits = []

    for idx in range(dataset.shape[0]):
        row = dataset.iloc[idx]
        video_path = os.sep.join([DATASET_DIR, "videos", row["video_name"]])
        data_path = os.sep.join([DATASET_DIR, "pose_data", row["video_name"] + ".pkl"])

        if "shoot" in row["label"] or (not os.path.exists(data_path)) or (not os.path.exists(video_path)):
            omit += 1
            continue

        frame_n = 0
        cap = cv2.VideoCapture(video_path)
        with open(data_path, 'rb') as f:
            video_data = pickle.load(f)

        tracker_1 = Tracker()
        tracker_2 = Tracker()
        tracker_rand = Tracker()
        num_samples = len(samples)
        for frame_data in video_data:
            frame_n = frame_data["frame_n"]

            poses = PoseEstimator.filter(frame_data["poses"])
            if frame_n % 2 == 0:
                samples += tracker_1.update_tracks(None, frame_n, poses)

            if (frame_n + 1) % 2 == 0:
                samples += tracker_2.update_tracks(None, frame_n, poses)

            if np.random.rand() > 0.6:
                samples += tracker_rand.update_tracks(None, frame_n, poses)

        num_samples = len(samples) - num_samples

        labels += [row["label"]] * num_samples
        splits += [row["split"]] * num_samples

        if idx % 5 == 0:
            print_progress(idx + 1, omit, dataset.shape[0], start_time)

    print()
