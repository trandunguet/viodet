import numpy
import cv2 as cv


class PreProcess:
    def __init__(self):
        # constants----------------
        self.FRAME_RATE = 25  # 25 frames per second
        self.MOVEMENT_INTERVAL = 3  # difference between considered frames
        self.N = 4  # number of vertical blocks per frame
        self.M = 4  # number of horizontal blocks per frame
        self.FRAME_GAP = 2 * self.MOVEMENT_INTERVAL
        self.SEQUENCE_LENGTH = 5
        # -------------------------
        self.total_frames = 0
        self.fps = 0
        self.time = 0
        # -------------------------
        self.dim = 100

    def read_video(self, video_name):
        self.cap = cv.VideoCapture(video_name)
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.time = self.total_frames / self.fps
        self.current_frame_id = 0

    def get_next_3_frames(self):
        frame_0 = self.get_frame_from_index(self.current_frame_id)
        self.current_frame_id += self.MOVEMENT_INTERVAL
        frame_1 = self.get_frame_from_index(self.current_frame_id)
        self.current_frame_id += self.MOVEMENT_INTERVAL
        frame_2 = self.get_frame_from_index(self.current_frame_id)
        return (frame_0, frame_1, frame_2)

    def get_next_sequence(self):
        next_sequence = []
        for i in range(self.SEQUENCE_LENGTH):
            next_sequence.append(self.get_next_3_frames())
        return next_sequence

    def get_frame_from_index(self, index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, img = self.cap.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = self.resize_frame(img)
        return img

    def get_original_frame_from_index(self, index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, img = self.cap.read()
        return img

    def resize_frame(self, frame):
        rescale = float(self.dim)/(frame.shape[1])
        if rescale < 0.8:
            dim = (self.dim, int(frame.shape[0] * rescale))
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        return frame

    def set_video_dimension(self, dim):
        self.dim = dim
