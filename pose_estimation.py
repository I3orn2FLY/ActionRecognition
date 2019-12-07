import cv2
import imutils
import numpy as np
import sys
import os

OPENPOSE_FOLDER = "/home/kenny/Workspace/Libraries/openpose"

sys.path.append(os.path.join(OPENPOSE_FOLDER, "build/python"))
from openpose import pyopenpose as op
from config import *


class PoseEstimator():
    needed_points = list(range(1, 9))
    def __init__(self, hand=False, face=False):
        params = dict()
        params["model_folder"] = os.path.join(OPENPOSE_FOLDER, "models")
        params["face"] = hand
        params["hand"] = face
        params["model_pose"] = "BODY_25"
        params["keypoint_scale"] = 3
        params["num_gpu"] = 1
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        self.idxs = list(range(NUM_POINTS))
        # self.needed_points = list(range(1, 11)) + list([12, 13])

    @staticmethod
    def filter(poses):
        filtered = []
        for pose in poses:

            if sum(pose[:, 2][PoseEstimator.needed_points] > 0.5) == len(PoseEstimator.needed_points):
                filtered.append(pose)
        if not filtered:
            return None

        return np.array(filtered)

    def estimate_pose(self, imageToProcess, filter=False, draw=False):
        datum = op.Datum()
        if not isinstance(imageToProcess, np.ndarray):
            return None
        datum.cvInputData = imageToProcess
        self.opWrapper.emplaceAndPop([datum])

        poses = datum.poseKeypoints
        if len(poses.shape) < 1:
            return None

        poses = poses[:, self.idxs]
        if filter:
            return self.filter(poses)

        return poses
