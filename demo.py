from tracker import Tracker
from pose_estimation import PoseEstimator

import cv2

if __name__ == "__main__":

    source = "demo.mp4"
    # source = "asd.avi"

    cap = cv2.VideoCapture(source)
    tracker = Tracker(model_name="LSTM")
    pose_estimator = PoseEstimator()

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_n % 2 == 0:
            poses = pose_estimator.estimate_pose(frame, filter=True)

            samples = tracker.update_tracks(frame, frame_n, poses, draw=True)
        frame_n += 1

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Window", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
