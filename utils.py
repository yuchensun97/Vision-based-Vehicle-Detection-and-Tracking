import cv2
import numpy as np
import matplotlib.pyplot as plt

from constant import *


def plot_sample(images, sample_size=SAMPLE_SIZE, save_path=None):
    indices = np.random.randint(0, high=len(images) - 1, size=sample_size)
    f, ax = plt.subplots(1, SAMPLE_SIZE, figsize=(3 * SAMPLE_SIZE, 3))
    for i, idx in enumerate(indices):
        image = images[idx]
        ax[i].imshow(image)
    if save_path is not None:
        plt.savefig(save_path)


def plot_image(image, save_path=None):
    plt.imshow(image)
    if save_path is not None:
        plt.savefig(save_path)


def plot_bbox(image, bboxes):
    new_image = image.copy()
    for bbox in bboxes:
        x, y, w, h = bbox
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(new_image, top_left, bottom_right, (0, 255, 0), 6)

    # plt.imshow(new_image)
    # plt.show()

    return new_image

def plot_cbox(image, bbox, color = (0, 255, 0)):
    '''
    plot a single box
    '''
    new_image = image.copy()
    x, y, w, h = bbox
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    new_image = cv2.rectangle(new_image, top_left, bottom_right, color, 6)

    return new_image


def plot_heatmap(img, bboxes):
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float64)
    for bbox in bboxes:
        x, y, w, h = bbox
        heatmap[y:y + h, x:x + w] = 255
    # plt.imshow(heatmap, cmap="gray")
    # plt.show()

    heatmap = heatmap.astype(np.uint8)
    ret, binary = cv2.threshold(heatmap, 40, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < BBOX_THRESH or h < BBOX_THRESH:
            continue
        new_bboxes.append((x, y, w, h))

    return new_bboxes
