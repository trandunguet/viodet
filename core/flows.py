import numpy as np


class Flows:
    def __init__(self, frames):
        self.frames = frames
        self.height = frames[0].shape[0]
        self.width = frames[0].shape[1]

    def get_vif(self):
        binary_sum = np.zeros((self.height, self.width))
        frame = self.frames[0]
        prev_magnitude = np.sqrt(np.square(frame[:,:,0]) + np.square(frame[:,:,1]))
        
        for frame in self.frames[1:]:
            magnitude = np.sqrt(np.square(frame[:,:,0]) + np.square(frame[:,:,1]))
            magnitude_change = abs(magnitude - prev_magnitude)
            threshold = np.mean(abs(magnitude_change))
            binary = np.where(magnitude_change < threshold, 0, 1)
            binary_sum += binary

            prev_magnitude = magnitude
            
        vif, _ = np.histogram(binary_sum, range(len(self.frames) + 1))
        return vif
