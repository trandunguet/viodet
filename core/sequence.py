import cv2 as cv
from core.flows import Flows


class Sequence:
    def __init__(self, frames, farnback_params):
        self.frames = frames
        self.height = frames[0].shape[0]
        self.width = frames[0].shape[1]
        self.farnback_params = farnback_params

    def get_flows(self):
        flows = []
        prev_frame = cv.cvtColor(self.frames[0], cv.COLOR_RGB2GRAY)

        for frame in self.frames[1:]:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, **self.farnback_params)
            flows.append(flow)

            prev_frame = frame

        return Flows(flows)
