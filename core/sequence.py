import cv2 as cv
from core.flows import Flows


class Sequence:
    def __init__(self, frames):
        self.frames = frames
        self.height = frames[0].shape[0]
        self.width = frames[0].shape[1]
        self.farnback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def get_flows(self):
        flows = []
        prev_frame = cv.cvtColor(self.frames[0], cv.COLOR_RGB2GRAY)

        for frame in self.frames[1:]:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            flow = cv.calcOpticalFlowFarneback(prev_frame, frame, None, **self.farnback_params)
            flows.append(flow)

            prev_frame = frame

        return Flows(flows)
