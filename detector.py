import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import imageio
import os
import pickle

from utils import *
from infer import find_cars


def objectTracking(raw_video):
    model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

    cap = cv2.VideoCapture(raw_video)
    images = []
    frame_cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        print("frame:", frame_cnt)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes = []
        for scale in [1, 1.5, 2.0]:
            curr_bboxes = find_cars(frame, model, scaler, scale=scale)
            bboxes.extend(curr_bboxes)

        bboxes = plot_heatmap(frame, bboxes)
        vis = plot_bbox(frame, bboxes)

        images.append(img_as_ubyte(vis))

    print("Saving gif...")
    imageio.mimsave('export/{}.gif'.format(frame_cnt), images)
    print("Finish tracking")


if __name__ == "__main__":
    objectTracking("data/project_video.mp4")
