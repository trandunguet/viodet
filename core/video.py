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

    def next_sequence(self):
        frames = []

        for _ in range(self.sequence_length):
            ready, frame = self.cap.read()
            if not ready:
                return None
            frames.append(frame)

        return Sequence(frames)

    def seek(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
