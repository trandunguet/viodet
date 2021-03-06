import cv2
import numpy as np
from .OpticalFlow import OptFlow
import math


class VioFlow:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.B_height = 0
        self.B_width = 0
        self.bins = np.arange(0.0, 1.05, 0.05, dtype=np.float64)

    def getViolentFlow(self, sequence):
        flow = OptFlow()
        temp_flows = []
        for (frame_0, frame_1, frame_2) in sequence:
            (vx1, vy1, w1) = flow.sorFlow(frame_0, frame_1)
            (vx2, vy2, w2) = flow.sorFlow(frame_1, frame_2)

            m1 = flow.getFlowMagnitude(vx1, vy1)
            m2 = flow.getFlowMagnitude(vx2, vy2)

            change_mag = abs(m2-m1)
            binary_mag = np.ones(change_mag.shape, dtype=np.float64)
            threshold = np.mean(change_mag, dtype=np.float64)
            temp_flows.append(np.where(change_mag < threshold, 0, binary_mag))

        flow_video = np.zeros(change_mag.shape, dtype=np.float64)
        for each_flow in temp_flows:
            flow_video = flow_video + each_flow

        flow_video = flow_video / len(temp_flows)

        self.height = flow_video.shape[0]
        self.width = flow_video.shape[1]
        self.B_height = int(math.floor((self.height - 11)/4))
        self.B_width = int(math.floor((self.width - 11)/4))

        return flow_video

    def histc(self, X, bins):
        map_to_bins = np.digitize(X, bins)
        r = np.zeros(bins.shape, dtype=np.float64)
        for i in map_to_bins:
            r[i-1] += 1
        return r

    def getBlockHist(self, flow_video):
        flow_vec = np.reshape(
            flow_video, (flow_video.shape[0]*flow_video.shape[1], 1))
        count_of_bins = self.histc(flow_vec, self.bins)
        return count_of_bins/np.sum(count_of_bins)

    def getFeatureVector(self, sequence):
        frame_hist = []
        flow_video = self.getViolentFlow(sequence)
        for y in range(6, self.height-self.B_height-4, self.B_height):
            for x in range(6, self.width-self.B_width-4, self.B_width):
                block_hist = self.getBlockHist(
                    flow_video[y:y+self.B_height, x:x+self.B_width])
                frame_hist = np.append(frame_hist, block_hist, axis=0)
        return frame_hist

    def writeFeatureToFile(self, sequence, filename):
        np.savetxt(filename, self.getFeatureVector(sequence), delimiter=',')
