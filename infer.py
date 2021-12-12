import numpy as np
import cv2
import os
import pickle
from torch.utils.data import DataLoader
import time

from constant import *
from utils import *
from dataset import VehicleDataset
from feature import FeatureExtractor


def find_cars(image, model, scaler, scale: float = 1):
    # Rescale image
    new_h = int(image.shape[0] / scale)
    new_w = int(image.shape[1] / scale)
    new_image = cv2.resize(image, (new_w, new_h))

    x_steps = (new_image.shape[1] - WINDOW_SIZE) // PIXEL_PER_STEP
    y_steps = (new_image.shape[0] - WINDOW_SIZE) // PIXEL_PER_STEP

    feature_extractor = FeatureExtractor()
    bboxes = []
    for xb in range(x_steps):
        if xb < x_steps // 2:
            continue
        for yb in range(y_steps):
            if yb < y_steps // 2:
                continue
            xr = xb * PIXEL_PER_STEP
            yr = yb * PIXEL_PER_STEP

            # Extract the image patch
            sub_image = new_image[yr:yr + WINDOW_SIZE, xr:xr + WINDOW_SIZE]
            # print(xr, yr)
            # plot_image(sub_image, "data/aaa/" + str(xr) + "," + str(yr) + ".png")
            features = feature_extractor.get_features([sub_image])

            # Scale features and make a prediction
            test_features = scaler.transform(features)
            test_prediction = model.predict(test_features)[0]
            # print(test_prediction)

            if test_prediction == 1:
                x = int(xr * scale)
                y = int(yr * scale)
                w = int(WINDOW_SIZE * scale)
                h = int(WINDOW_SIZE * scale)
                bbox = (x, y, w, h)
                bboxes.append(bbox)

    return bboxes


if __name__ == "__main__":
    test_model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
    test_scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

    test_dataset = VehicleDataset(TEST_DATA_DIR, label=-1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for data in test_loader:
        images, labels = data
        test_image = images[0].numpy()

        start_time = time.time()
        bbox_pyramids = []
        for scale in [1, 1.5, 2.0]:
            bboxes = find_cars(test_image, test_model, test_scaler, scale=scale)
            plot_bbox(test_image, bboxes)
            bbox_pyramids.extend(bboxes)

        new_bboxes = plot_heatmap(test_image, bbox_pyramids)
        plot_bbox(test_image, new_bboxes)
        print(time.time() - start_time)
        break
