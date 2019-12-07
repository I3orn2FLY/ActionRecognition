import numpy as np
import glob
import sys
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import Counter

sys.path.append("../")
from config import *
from pose_estimation import PoseEstimator
from tracker import Tracker


def cm_analysis(y_true, y_pred, labels, filename, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=None)
    print(cm)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    # plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # ls = list(glob.glob(KTH_FOLDER + "/**/*.avi", recursive=True))
    # pose_estimator = PoseEstimator()
    #
    # tracker = Tracker(model_name="LSTM")
    # correct = 0
    # L = len(ls)
    # preds = []
    # gts = []
    # for video_idx, video_file in enumerate(ls):
    #     label = video_file.split("/")[-2]
    #     gts.append(label)
    #     cap = cv2.VideoCapture(video_file)
    #     frame_n = 0
    #     predictions = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: break
    #
    #         if frame_n % 2 == 0:
    #             poses = pose_estimator.estimate_pose(frame)
    #             tracker.update_tracks(frame, frame_n, poses, draw=True)
    #
    #             predictions += tracker.get_actions()
    #
    #         frame_n += 1
    #     try:
    #         prediction = Counter(predictions).most_common(1)[0][0]
    #     except:
    #         prediction = label
    #
    #     preds.append(prediction)
    #
    #     print("\r" + str(video_idx + 1), "out of", L, "done", end="")
    #     sys.stdout.flush()

    # print()
    # print("Results. Correct: " + str(correct) + "/" + str(L), "Accuracy: %.2f" % (correct / L * 100))
    #
    # with open("../vars/gts", 'wb') as f:
    #     pickle.dump(gts, f)
    #
    # with open("../vars/preds", 'wb') as f:
    #     pickle.dump(preds, f)

    with open("../vars/gts", 'rb') as f:
        gts = pickle.load(f)

    with open("../vars/preds", 'rb') as f:
        preds = pickle.load(f)

    # print(pd.Series(gts).value_counts())
    # print(pd.Series(preds).value_counts())

    classes = ["walking", "boxing", "jogging", "running", "handclapping", "handwaving"]
    LABEL_ENCODER = {label: idx for idx, label in enumerate(classes)}
    LABEL_DECODER = {idx: label for idx, label in enumerate(classes)}

    print(sum(np.array(gts) == np.array(preds))/len(gts))
    # gts = [LABEL_ENCODER[gt] for gt in gts]
    # preds = [LABEL_ENCODER[pred] for pred in preds]
    assert len(gts) == len(preds)
    cm_analysis(gts, preds, classes, filename="confusion_matrix.png")
