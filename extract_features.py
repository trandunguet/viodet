#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import core

if __name__ == '__main__':
    right_arrow = 83
    space = 32
    escape = 27
    n_key = 110
    p_key = 112

    video = core.Video("assets/test.avi")
    plt.ion()
    plt.show()

    output_negative = open("features/negative.txt", "a+")
    output_positive = open("features/positive.txt", "a+")

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

            if key == n_key:
                output_negative.write(np.array2string(vif) + '\n')
                break

            if key == p_key:
                output_positive.write(np.array2string(vif) + '\n')
                break

            key = cv.waitKey()

    output_negative.close()
    output_positive.close()
