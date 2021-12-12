import numpy as np
import cv2
from numpy.linalg import inv
from scipy.linalg import block_diag

from ast import literal_eval

from constant import *
import json

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
from utils import plot_bbox

from tracker import EKF
from iou import overlap, update_track_object

def read_detections(filename):
    detections = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            line = literal_eval(line)
            detections.append(line)
    return detections

def smooth(detections, iou_ratio):
    # access tracking object
    track_object = update_track_object(detections, iou_ratio)
    track_EKF = track_object.copy()

    for i, t in enumerate(track_object):
        t_boxes = t['boxes']
        new_boxes = []
        # create new EKF tracker
        x, y, w, h = t_boxes[0]
        init_state = np.array([x, 0, y, 0, w, 0, h, 0])
        tracker = EKF(init_state, fps = 25.2)
        tracker.update_no_measurement()
        new_state, _ = tracker.retrieve()
        new_box = new_state[0:-1:2].tolist()
        new_boxes.append(new_box)

        # update new EKF tracker
        for box in t_boxes[1:]:
            zx, zy, zw, zh = box
            z = np.array([zx, zy, zw, zh])
            tracker.update(z)
            new_state, _ = tracker.retrieve()
            new_box = new_state[0:-1:2].tolist()
            new_boxes.append(new_box)

        # replace boxes in each object dictionary
        # with EKF updated boxes
        track_EKF[i]['boxes'] = new_boxes

    return track_EKF

def visualization(boxes_file, video):
    '''
    plot objects' bboxes in the video and save the video
    Args:
        boxes_list: list of dictionaray [{'boxes': shape(*, 4), 'start frame': scaler}, {...}, ...]
        video: video path
    return:
        none
    '''
    # TODO: fill this
    cap = cv2.VideoCapture(video)
    
    dict_obj = {}
    file1 = open(boxes_file, "r")
    for _ in open(boxes_file):
        line = file1.readline()
        s = line.find("start frame")
        end = line.find("}")
        start_frame = int(line[s + 13: end])
        boxes = line[12: s - 5]

        boxes_split = boxes.split("], [")
        final_boxes_split = []
        for box in boxes_split:
            box_split = box.split(", ")
            temp = []
            for i in box_split:
                temp.append(int(i))
            final_boxes_split.append(temp)
        dict_obj[start_frame] = final_boxes_split
    dict_obj_final = {}
    for key in dict_obj:
        len_temp = len(dict_obj[key])
        for i in range(len_temp):
            if key + i in dict_obj_final:
                dict_obj_final[key + i].append(tuple(dict_obj[key][i]))
            else:
                dict_obj_final[key + i] = [tuple(dict_obj[key][i])]
    
    frame_cnt = 0
    images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        print("frame:", frame_cnt)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes = []
        if frame_cnt - 1 in dict_obj_final:
            bboxes = dict_obj_final[frame_cnt - 1]
        vis = plot_bbox(frame, bboxes)

        images.append(img_as_ubyte(vis))
    

    print("Saving gif...")
    imageio.mimsave('export/{}.gif'.format(frame_cnt), images)
    print("Finish tracking")

if __name__ == "__main__":
    # initialize
    f = 'export/file.txt'
    v = 'data/project_video.mp4'
    iou_ratio = 0.15
    detections = read_detections(f)
    track_EKF = smooth(detections, iou_ratio)

    with open('export/object_ekf.txt', 'w') as new_file:
        for t in track_EKF:
            json.dump(t, new_file)
            new_file.write('\n')

    new_file.close()

    visualization('export/object_ekf.txt', v)