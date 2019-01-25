import cv2 as cv
import numpy as np
import math

class VideoProcess:
    def __init__(self, video_name):
        self.cap = cv.VideoCapture(video_name)
        self.SEQUENCE_LENGTH = 10
        self.HEIGHT = 240
        self.WIDTH = 320
        self.sequence = []
        self.flows = []
        self.farnback_params = dict(
            pyr_scale = 0.5,
            levels = 3,
            winsize = 15,
            iterations = 3,
            poly_n = 5,
            poly_sigma = 1.2,
            flags = 0,
        )

    def process_next_sequence(self):
        self.sequence.clear()

        for _ in range(self.SEQUENCE_LENGTH):
            ready, frame = self.cap.read()
            if ready:
                self.sequence.append(frame)
        self.extract_vif()

    def extract_vif(self):
        if len(self.sequence) != self.SEQUENCE_LENGTH:
            return

        self.flows.clear()
        binary_sum = np.zeros((self.HEIGHT, self.WIDTH))

        prev_frame = cv.cvtColor(self.sequence[0], cv.COLOR_RGB2GRAY)
        frame = cv.cvtColor(self.sequence[1], cv.COLOR_RGB2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_frame,frame, None, **self.farnback_params)
        prev_magnitude = np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1]))

        for frame in self.sequence[2:]:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, **self.farnback_params)
            self.flows.append(flow)

            magnitude = np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1]))
            magnitude_change = abs(magnitude - prev_magnitude)
            threshold = np.mean(abs(magnitude_change))
            binary = np.where(magnitude_change < threshold, 0, 1)
            binary_sum += binary

            prev_frame = frame
            prev_magnitude = magnitude

        self.vif, _ = np.histogram(binary_sum, range(self.SEQUENCE_LENGTH))
