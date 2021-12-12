'''
Combining detector and tracker
tracker can track multiple object in the video
'''
import numpy as np
import json
from utils import *
from constant import *

from constant import t_min

from ast import literal_eval

def overlap(box1, box2):
    '''
    check the overlap of two boxes
    @param
    box: shape(4,) ndarray, [x, y, w, h]

    @return
    ratio: intersection / union
    '''
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # compute each box's area
    s1 = w1 * h1
    s2 = w2 * h2

    # compute intersection
    left = min(x1, x2)
    right = max(x1 + w1, x2 + w2)
    w = w1 + w2 - (right - left)

    up = min(y1, y2)
    down = max(y1 + h1, y2 + h2)
    h = h1 + h2 - (down - up)

    if w <= 0 or h <= 0:
        return 0
    else:
        intersect = w * h
        union = s1 + s2 - intersect
        ratio = intersect / union
        return ratio


def update_track_object(detections, iou_ratio):
    '''
    IOU tracker
    https://ieeexplore.ieee.org/document/8078516

    @param:
    detections: list of detections per frame
    iou_ratio: iou threshold

    @return:
    track_finished: list of track
    '''
    tracks_active  = []
    tracks_finished = []
    for frame_idx, dets_per_frame in enumerate(detections, start = 1):
        for trk_a in tracks_active:
            if len(dets_per_frame) > 0:
                iou = [overlap(trk_a['boxes'][-1], det) for det in dets_per_frame]
                iou = np.array(iou)
                det_best_idx = np.argmax(iou)
                det_best = dets_per_frame[det_best_idx]
                if overlap(trk_a['boxes'][-1], det_best) >= iou_ratio:
                    # add d_best to t_i
                    trk_a['boxes'].append(det_best)
                    # remove d_best from dets_per_frame
                    dets_per_frame.pop(det_best_idx)
            else:
                # finish track when min # of matches is hit
                if len(trk_a['boxes']) >= t_min:
                    tracks_finished.append(trk_a)
                tracks_active.remove(trk_a)
        
        # start new track with new detection and insert into tracks_active
        new_tracks = [{'boxes': [det], 'start frame': frame_idx} for det in dets_per_frame]
        tracks_active += new_tracks
    
    # finish all remaining active tracks
    tracks_finished += [trk_a for trk_a in tracks_active if len(trk_a['boxes']) >= t_min]

    return tracks_finished

if __name__ == "__main__":
    # read detections
    detections = []
    with open('export/file.txt') as f:
        for line in f:
            line = line.rstrip()
            line = literal_eval(line)
            detections.append(line)
    
    track_object = update_track_object(detections, 0.15)

    with open('export/object.txt', 'w') as f:
        for t in track_object:
            json.dump(t, f)
            f.write('\n')
    
