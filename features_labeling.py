#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import sys
import time

import core

if __name__ == '__main__':
    start_time = time.clock()

    # Read label file, mark fighting frames ids
    video_name = "clip1.wmv"
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
    video = core.Video("assets/{}".format(video_name))
    output_negative = open("features/negative.txt", "a+")
    output_negative_ref = open("features/negative_ref.txt", "a+")
    output_positive = open("features/positive.txt", "a+")
    output_positive_ref = open("features/positive_ref.txt", "a+")

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
            output_positive_ref.write("{} {} {}\n".format(video_name, sequence.id_range[0], sequence.id_range[1]))
            positive_count += 1
        else:
            output_negative.write(np.array2string(vif) + '\n')
            output_negative_ref.write("{} {} {}\n".format(video_name, sequence.id_range[0], sequence.id_range[1]))
            negative_count += 1

        percentage = int(sequence.id_range[0] * 100 / video.frame_count)
        sys.stdout.write('\rExtracting & labeling features: {} %'.format(percentage))
        sys.stdout.flush() 

    output_negative.close()
    output_positive.close()

    print("Extrated total: {} sequences".format(negative_count + positive_count))
    print("Positive: {}".format(positive_count))
    print("Negative: {}".format(negative_count))
    print("Time elapsed: {} seconds".format(time.clock() - start_time))
