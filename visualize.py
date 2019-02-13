#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import core

if __name__ == '__main__':
    right_arrow = 83
    space = 32
    escape = 27

    video = core.Video("assets/test.avi")
    plt.ion()
    plt.show()

    while True:
        sequence = video.next_sequence()
        if not sequence:
            print("Video ended.")
            break

        vif, mean, variance = sequence.get_flows().get_vif()
        print("vif: ", vif)
        print("mean: ", mean)
        print("variance: ", variance, "\n")
        plt.clf()
        plt.plot(vif)
        plt.draw()
        plt.pause(0.001)

        key = space
        while key != right_arrow:
            if key == space:
                for frame in sequence.frames:
                    cv.imshow("video", frame)
                    cv.waitKey(30)

            if key == escape:
                exit()

            key = cv.waitKey()
