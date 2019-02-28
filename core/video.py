# 3rd party
import cv2 as cv
import numpy as np


class Video:
    """
    Wrapper for OpenCV's VideoCapture class
    """

    def __init__(self, video_name):
        self.cap = cv.VideoCapture(video_name)
        if not self.cap.isOpened():
            raise Exception('OPEN VIDEO FAILED!')

        self.sequence_length = 10
        self.height = 240
        self.width = 320
        self.frame_count = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.farnback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        _, self.prev_frame = self.next_frame()
        self.sequence = []


    def __next_sequence_flows(self):
        if len(self.sequence) > 0:
            self.sequence.pop(0)

        while len(self.sequence) < self.sequence_length:
            ready, frame = self.next_frame()
            if not ready:
                raise Exception("Video ended!")

            flow = cv.calcOpticalFlowFarneback(self.prev_frame, frame, None, **self.farnback_params)
            self.prev_frame = frame
            self.sequence.append(flow)


    def next_sequence_features(self):
        """
        Extract ViF from flows sequence
        """

        self.__next_sequence_flows()
        binary_sum = np.zeros((self.height, self.width))
        frame = self.sequence[0]
        prev_magnitude = np.sqrt(np.square(frame[:,:,0]) + np.square(frame[:,:,1]))
        
        for frame in self.sequence[1:]:
            magnitude = np.sqrt(np.square(frame[:,:,0]) + np.square(frame[:,:,1]))
            magnitude_change = abs(magnitude - prev_magnitude)
            threshold = np.mean(abs(magnitude_change))
            binary = np.where(magnitude_change < threshold, 0, 1)
            binary_sum += binary

            prev_magnitude = magnitude
            
        vif, _ = np.histogram(binary_sum, range(len(self.sequence) + 1))
        # mean = np.mean(binary_sum)
        # variance = np.var(binary_sum)
        return vif


    def seek(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)


    def get_current_frame_id(self):
        return self.cap.get(cv.CAP_PROP_POS_FRAMES)


    def next_frame(self):
        ready, frame = self.cap.read()
        if not ready:
            return False, None

        frame = cv.resize(frame, (self.width, self.height))
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        return ready, frame
