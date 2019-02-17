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
        self.frame_count = self.cap.get(cv.CAP_PROP_FRAME_COUNT)

    def next_sequence(self):
        frames = []
        current_frame_id = self.cap.get(cv.CAP_PROP_POS_FRAMES)

        for _ in range(self.sequence_length):
            ready, frame = self.cap.read()
            if not ready:
                return None
            
            frame = cv.resize(frame, (self.width, self.height))
            frames.append(frame)

        return Sequence(frames, (current_frame_id, current_frame_id + self.sequence_length - 1))

    def seek(self, frame_id):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
