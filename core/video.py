import cv2 as cv
from core.sequence import Sequence


class Video:
    def __init__(self, video_name):
        self.cap = cv.VideoCapture(video_name)
        if not self.cap.isOpened():
            raise Exception('OPEN VIDEO FAILED!')

        self.sequence_length = 10
        self.height = 240
        self.width = 320
        self.farnback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def next_sequence(self):
        frames = []

        for _ in range(self.sequence_length):
            _, frame = self.cap.read()
            frames.append(frame)

        if len(frames) != self.sequence_length:
            raise Exception("VIDEO ENDED")

        return Sequence(frames, self.farnback_params)
