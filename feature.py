import numpy as np
import cv2
from skimage.feature import hog

from constant import *


class FeatureExtractor:
    def __init__(self, orient=ORIENTATION, pix_per_cell=PIXEL_PER_CELL, cell_per_block=CELL_PER_BLOCK,
                 spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS):
        self.orient = orient
        self.pix_per_cell = (pix_per_cell, pix_per_cell)
        self.cell_per_block = (cell_per_block, cell_per_block)
        self.spatial_size = (spatial_size, spatial_size)
        self.hist_bins = hist_bins

    def get_features(self, images):
        features = []
        for image in images:
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

            # Get 3 kinds of feature vectors
            spatial_features = self.__get_bin_spatial(feature_image)
            hist_features = self.__get_color_hist(feature_image)
            hog_features = self.__get_hog_features(feature_image)

            # Concatenate all feature vectors
            features.append(np.concatenate([spatial_features, hist_features, hog_features]))

        return features

    def __get_hog_features(self, image):
        features = []
        for channel in range(image.shape[2]):
            feature, _ = hog(image[:, :, channel],
                             orientations=self.orient,
                             pixels_per_cell=self.pix_per_cell,
                             cells_per_block=self.cell_per_block,
                             visualize=True)
            features.append(feature)

        return np.ravel(features)

    def __get_bin_spatial(self, image):
        features = cv2.resize(image, self.spatial_size).ravel()

        return features

    def __get_color_hist(self, image):
        features = []
        for channel in range(image.shape[2]):
            hist, _ = np.histogram(image[:, :, channel], bins=self.hist_bins, range=(0, 256))
            features.append(hist)

        return np.concatenate(features)

