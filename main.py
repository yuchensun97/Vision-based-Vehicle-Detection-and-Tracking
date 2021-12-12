import os

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from dataset import VehicleDataset
from feature import FeatureExtractor
from train import train, test
from utils import plot_sample
from constant import *

if __name__ == '__main__':
    # Initialize output directory
    for d in [EXPORT_DIR, MODEL_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Initialize dataset and dataloader
    vehicle_dataset = VehicleDataset(VEHICLES_DATA_DIR, label=1)
    vehicle_data_loader = DataLoader(vehicle_dataset, batch_size=BATCH_SIZE, shuffle=True)

    non_vehicle_dataset = VehicleDataset(NON_VEHICLES_DATA_DIR, label=0)
    non_vehicle_data_loader = DataLoader(non_vehicle_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Plot several sample data
    # sample_vehicle = []
    # for data in vehicle_data_loader:
    #     images, labels = data
    #     sample_vehicle.append(images.numpy())
    # plot_sample(np.concatenate(sample_vehicle), save_path=os.path.join(EXPORT_DIR, "vehicle_sample.png"))
    #
    # sample_non_vehicle = []
    # for data in non_vehicle_data_loader:
    #     images, labels = data
    #     sample_non_vehicle.append(images.numpy())
    # plot_sample(np.concatenate(sample_non_vehicle), save_path=os.path.join(EXPORT_DIR, "non_vehicle_sample.png"))

    # Extract features
    feature_extractor = FeatureExtractor()

    X, y = [], []
    for data in tqdm(vehicle_data_loader, position=0):
        images, labels = data
        X.extend(feature_extractor.get_features(images.numpy()))
        y.extend(labels.numpy().tolist())

    for data in tqdm(non_vehicle_data_loader, position=0):
        images, labels = data
        X.extend(feature_extractor.get_features(images.numpy()))
        y.extend(labels.numpy().tolist())

    X, y = np.array(X), np.array(y)

    train(X, y)

