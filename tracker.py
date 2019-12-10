import cv2
import torch
import numpy as np
from scipy.spatial import distance_matrix

from utils import get_model, predict
from config import *


class Track:
    def __init__(self):
        self.features = []
        self.track_points = []
        self.action = None

    def add_point(self, point, pose, frame_n):
        self.track_points.append(point)
        feat = np.zeros(NUM_FEAT)
        feat[0] = frame_n
        feat[1:] = pose[:, :2].reshape(-1)
        self.features.append(feat)

    def _get_action_crit(self):
        feats = np.array(self.features)[:, 1:].reshape(-1, NUM_POINTS, 2)
        height = np.mean(feats[:, 8, 1]) - np.mean(feats[:, 1, 1])
        for i in range(feats.shape[0]):
            feats[i] = feats[i] - feats[i, 1, :]
        if height < 0: return 0
        path = 0
        for i in range(1, feats.shape[0]):
            diff = feats[i] - feats[i - 1]
            norm = np.sum(np.linalg.norm(diff, axis=1))
            path += norm

        criterion = path / (SEQ_LENGTH - 1) / height

        if criterion > 5:
            return 0
        return criterion

    def set_action(self, action):
        self.action = action

    def get_track(self):
        return self.track_points

    def last_point(self):
        return self.track_points[-1]

    def length(self):
        return len(self.track_points)

    def last_frame_n(self):
        return self.features[-1][0]

    def get_sample(self):
        if len(self.features) < SEQ_LENGTH:
            return None
        elif len(self.features) == SEQ_LENGTH:
            action_crit = self._get_action_crit()

            # if action_crit < CRITERION_TH:
            #     return None

            feat_sample = np.array(self.features)
            feat_sample[:, 0] -= feat_sample[0, 0]
            del self.features[0]
            del self.track_points[0]
            return feat_sample


class Tracker:

    def __init__(self, model_name=None):
        self.tracks = []
        self.model = None
        if model_name is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = get_model(model_name, device)
            model.eval()
            self.model = model
            self.device = device

    def update_tracks(self, frame, frame_n, poses, draw=False):
        self.delete_old_tracks(frame_n)
        if poses is None: return []

        cands = [pose[8, :2] for pose in poses]
        if not self.tracks:
            self.tracks = [Track() for _ in poses]
            for idx, track in enumerate(self.tracks):
                track.add_point(cands[idx], poses[idx], frame_n)
            return []

        last_points = [track.last_point() for track in self.tracks]
        dist_mat = distance_matrix(cands, last_points)

        cands2tracks_dist = np.min(dist_mat, axis=1)
        track_idxs_dist = np.argmin(dist_mat, axis=1)
        tracks2cands_dist = np.min(dist_mat, axis=0)

        for cand_idx in range(len(cands)):
            pose = poses[cand_idx]
            cand2track_dist = cands2tracks_dist[cand_idx]
            track_idx_dist = track_idxs_dist[cand_idx]
            track2cand_dist = tracks2cands_dist[track_idx_dist]

            if cand2track_dist == track2cand_dist and cand2track_dist < DIST_TH:
                self.tracks[track_idx_dist].add_point(cands[cand_idx], pose, frame_n)
            else:
                track = Track()
                track.add_point(cands[cand_idx], pose, frame_n)
                self.tracks.append(track)

        samples = []

        tracks_with_actions = []
        for track in self.tracks:
            sample = track.get_sample()
            if sample is not None:
                samples.append(sample)
                tracks_with_actions.append(track)

        if self.model is not None and samples:
            inp = np.array(samples)

            with torch.no_grad():
                preds = predict(self.model, inp, self.device)
                if draw:
                    for pred, track in zip(preds, tracks_with_actions):
                        action = idx2label[pred]
                        track.set_action(action)

        if draw:
            self.draw_poses(poses, frame)
            self.draw_tracks(frame)
        return samples

    def get_actions(self):
        actions = []
        for track in self.tracks:
            if track.action is not None:
                actions.append(track.action)

        return actions

    def draw_poses(self, poses, frame):
        for pose in poses:
            coords = pose[:, :2] * frame.shape[:2][::-1]
            coords = coords.astype("int")

            for p in coords:
                cv2.circle(frame, (p[0], p[1]), int(frame.shape[0] / 240), (255, 255, 100), -1)

    def draw_tracks(self, frame):
        lines = []
        for tr in self.tracks:
            track = tr.track_points
            line = []

            for point in track:
                p = point * frame.shape[:2][::-1]
                line.append(p.astype("int"))
            lines.append(np.array(line))

            if tr.action is not None:
                p = tr.last_point() * frame.shape[:2][::-1]
                p = p.astype("int")
                cv2.putText(frame, tr.action, (p[0], p[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

        cv2.polylines(frame, lines, False, (0, 0, 255))

    def delete_old_tracks(self, frame_n):
        track_idx = 0
        while (track_idx < len(self.tracks)):
            if frame_n - self.tracks[track_idx].last_frame_n() > FRAME_TH:
                del self.tracks[track_idx]
            else:
                track_idx += 1
