'''
EKF filter for tracking
'''

import numpy as np
import cv2
from numpy.linalg import inv
from scipy.linalg import block_diag

from ast import literal_eval

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
from utils import plot_cbox

class EKF:
    def __init__(self, state, fps = 1, P_scaler = 1, Q_scaler = 0.625):
        '''
        @param:
        state -- 8x1 vector represents the frame state
        fps -- video fps
        P_scaler -- scaler for covariance matrix
        Q_scaler -- scaler for measurement noise
        '''
        # initial state
        # x = [x, x_dot, y, y_dot, w, w_dot, h, h_dot]
        # updated value
        self.state = state

        # time interval for two concective frame
        self.dt = 1 / fps 

        # initial state transition matrix
        self.A = np.eye(8)
        row = np.arange(0, 7, 2)
        col = np.arange(1, 8, 2)
        self.A[row, col] = self.dt

        # initial measurement noise term
        self.R_scaler = np.array([[self.dt**4/4., self.dt**3/2.],
                             [self.dt**3/2., self.dt**2.]])
        self.R = block_diag(self.R_scaler, self.R_scaler, self.R_scaler, self.R_scaler)


        # initial measurement matrix
        self.C = np.zeros((4, 8))
        row = np.arange(4)
        col = np.arange(0, 8, 2)
        self.C[row, col] = 1

        # initial state covariance matrix, 8X8
        # updated value
        self.P = np.diag(P_scaler * np.ones(8))

        # initial measurement noise covariance matrix, 4X4
        self.Q = np.diag(Q_scaler * np.ones(4))

    def update(self, z):
        '''
        update state x and covariance matrix P
        @param
        z -- input measurement
        '''
        # update mean of state and covariance
        x_mean = self.A @ self.state
        P_mean = self.A @ self.P @ self.A.T + self.R

        # update state and covariance
        self.K = P_mean @ self.C.T @ inv(self.C @ P_mean @ self.C.T + self.Q.T)
        x = x_mean + self.K @ (z - self.C @ x_mean)
        self.P = (np.eye(8) - self.K @ self.C) @ P_mean
        self.state = x.astype(int) # convert to integer

    def update_no_measurement(self):
        '''
        only for new detections
        '''
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.R
        self.state = self.state.astype(int)

    def retrieve(self):
        '''
        retrieve current EKF predicted state and covariance matrix
        '''
        return self.state, self.P

# for debugging
def image_EKF():
    # create ekf
    init_state = np.array([1050, 0, 390, 0, 228, 0, 123, 0])
    init_box = init_state[0:-1:2]
    P_scaler = 10
    Q_scaler = 10 / 16
    fps = 1
    tracker = EKF(init_state, fps, P_scaler, Q_scaler)

    # update state
    z = np.array([1022, 399, 234, 105])
    tracker.update(z)
    new_state, _ = tracker.retrieve()
    new_box = new_state[0:-1:2]

    #visualization
    img = plt.imread('export/frame004.jpg')
    plt.figure(figsize=(8, 20))
    img = plot_cbox(img, init_box, color=(0, 255, 0))
    ax = plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.title('Last updated: ' + str(init_box))

    img = plot_cbox(img, z, color=(255, 0, 0))
    ax = plt.subplot(3, 1, 2)
    plt.imshow(img)
    plt.title('Current measured:' + str(z))

    img = plot_cbox(img, new_box, color=(0, 0, 255))
    ax = plt.subplot(3, 1, 3)
    plt.imshow(img)
    plt.title('Updated: ' + str(new_box))

    plt.savefig('export/ekf_result.png')
    plt.show()

def video_EKF():
    # read data from the file
    detections = []
    with open('export/file.txt') as f:
        for line in f:
            line = line.rstrip()
            line = literal_eval(line)
            detections.append(line)

    # create new EKF tracker
    detections = detections[253:317]
    x, y, w, h = detections[0][0]
    init_state = np.array([x, 0, y, 0, w, 0, h, 0])
    traker = EKF(init_state, fps = 25.2)
    traker.update_no_measurement()
    new_state, _ = traker.retrieve()
    new_box = new_state[0:-1:2]
    
    cap = cv2.VideoCapture('data/project_video.mp4')
    images = []
    frame_cnt = 0

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_cnt < 253:
            frame_cnt += 1
            continue

        if frame_cnt >= 316:
            break

        print("frame:", frame_cnt)

        frame_cnt += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vis = plot_cbox(frame, new_box)
        
        zx, zy, zw, zh = detections[i][0]
        z = np.array([zx, zy, zw, zh])
        traker.update(z)
        new_state, _ = traker.retrieve()
        new_box = new_state[0:-1:2]
        i += 1
        images.append(img_as_ubyte(vis))

    print("Saving gif...")
    imageio.mimsave('export/{}.gif'.format(frame_cnt), images)
    print("Finish tracking")
    
    

# for debugging
if __name__ == "__main__":
    video_EKF()