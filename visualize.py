#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import core

if __name__ == '__main__':
    video = core.Video("assets/test.avi")
    vif, mean, variance = video.next_sequence().get_flows().get_vif()
    print("vif: ", vif)
    print("mean: ", mean)
    print("variance: ", variance)
