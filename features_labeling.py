#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import sys
import time

import core

if __name__ == '__main__':
    start_time = time.clock()

    # Read label file, mark fighting frames ids
    markup_file = open("assets/clip1_label.txt", "r")
    markup_file.readline()
    fight_frames = []

    while True:
        line = markup_file.readline().replace('\t', ' ').replace(';', ' ').replace('\n', ' ').split()
        if len(line) < 3:
            break

        if line[0][0] == '[':
            line.pop(0)

        if line[0][0] == '[':
            line.pop(0)

        if line[2] != 'Fight':
            continue
        
        fight_frames.append((int(line[0]), int(line[1])))

    # Read video, extract features, label features
    video = core.Video("assets/clip1.wmv")
    output_negative = open("features/negative.txt", "a+")
    output_positive = open("features/positive.txt", "a+")
    negative_count = positive_count = 0

    def isViolent(id_range):
        for begin, end in fight_frames:
            if begin <= id_range[0] < id_range[1] <= end:
                return True
        return False

    while True:
        sequence = video.next_sequence()
        if not sequence:
            print("\nVideo ended.")
            break

        vif, _, _ = sequence.get_flows().get_vif()

        if isViolent(sequence.id_range):
            output_positive.write(np.array2string(vif) + '\n')
            positive_count += 1
        else:
            output_negative.write(np.array2string(vif) + '\n')
            negative_count += 1

        percentage = int(sequence.id_range[0] * 100 / video.frame_count)
        sys.stdout.write('\rextracting & labeling features: {} %'.format(percentage))
        sys.stdout.flush() 

    output_negative.close()
    output_positive.close()

    print("extrated total: {} sequences".format(negative_count + positive_count))
    print("positive: {}".format(positive_count))
    print("negative: {}".format(negative_count))
    print("time elapsed: {} seconds".format(time.clock() - start_time))
