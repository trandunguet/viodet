#!/usr/bin/env python3

import sys
import time
import os

import cv2 as cv
import numpy as np

import core

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid argument!")
        print("Usage: {} VIDEO_PATH MARKUP_FILE_PATH OUTPUT_FOLDER_PATH".format(sys.argv[0]))
        exit()

    # Read label file, mark fighting frames ids
    video_path = sys.argv[1]
    markup_file_path = sys.argv[2]
    output_folder_path = sys.argv[3]

    start_time = time.clock()

    markup_file = open(markup_file_path, "r")
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
    video = core.Video(video_path)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_negative = open("{}/negative.txt".format(output_folder_path), "a+")
    output_negative_ref = open("{}/negative_ref.txt".format(output_folder_path), "a+")
    output_positive = open("{}/positive.txt".format(output_folder_path), "a+")
    output_positive_ref = open("{}/positive_ref.txt".format(output_folder_path), "a+")

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
            output_positive_ref.write("{} {} {}\n".format(video_path, sequence.id_range[0], sequence.id_range[1]))
            positive_count += 1
        else:
            output_negative.write(np.array2string(vif) + '\n')
            output_negative_ref.write("{} {} {}\n".format(video_path, sequence.id_range[0], sequence.id_range[1]))
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
