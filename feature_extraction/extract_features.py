import cv2
import numpy as np
import pickle
import sys
import os

import pandas as pd
import time

sys.path.append("../")

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


if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join(DATASET_DIR, "annotation.csv"))
    pose_estimator = PoseEstimator()
    start_time = time.time()
    omit = 0
    for idx in range(dataset.shape[0]):
        row = dataset.iloc[idx]
        label = row["label"]
        video_path = os.sep.join([DATASET_DIR, "videos", row["video_name"]])
        data_path = os.sep.join([DATASET_DIR, "pose_data", row["video_name"] + ".pkl"])

        if "shoot" in label or os.path.exists(data_path) or (not os.path.exists(video_path)):
            omit += 1
            continue

        frame_n = 0
        cap = cv2.VideoCapture(video_path)
        video_data = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            poses = pose_estimator.estimate_pose(frame, filter=False)
            if poses is None:
                continue

            frame_data = {"frame_n": frame_n, "poses": poses}
            video_data.append(frame_data)
            frame_n += 1

        cap.release()

        if video_data:
            with open(data_path, 'wb') as f:
                pickle.dump(video_data, f)

        print_progress(idx + 1, omit, dataset.shape[0], start_time)

    print()
