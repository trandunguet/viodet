import sys
import time
import os

import cv2 as cv
import numpy as np

from core import PreProcess

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Invalid argument!")
        print("Usage: {} VIDEO_PATH MARKUP_FILE_PATH OUTPUT_FOLDER_PATH".format(sys.argv[0]))
        exit()

    # Read label file, mark fighting frames ids
    cap = cv.VideoCapture(sys.argv[1])
    markup_file_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    tmp = PreProcess()
    sequence_length = tmp.SEQUENCE_LENGTH * tmp.FRAME_GAP

    start_time = time.clock()

    markup_file = open(markup_file_path, "r")
    markup = []

    for line in markup_file:
        words = line.split()
        markup.append((int(words[0]), int(words[1]), words[2]))

    output_path_negative = '{}/negative'.format(output_folder_path)
    output_path_positive = '{}/positive'.format(output_folder_path)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if not os.path.exists(output_path_negative):
        os.makedirs(output_path_negative)

    if not os.path.exists(output_path_positive):
        os.makedirs(output_path_positive)

    negative_list_file = open('{}/list.txt'.format(output_path_negative), 'w')
    positive_list_file = open('{}/list.txt'.format(output_path_positive), 'w')

    negative_count = positive_count = 0
    fourcc = cv.VideoWriter_fourcc(*'XVID')

    positive_id = 0
    negative_id = 0
    for line in markup:
        for start in range(line[0], line[1] - sequence_length, sequence_length):
            if line[2] == 'negative':
                negative_list_file.write('{}.avi\n'.format(negative_id))
                output_path = '{}/{}/{}.avi'.format(output_folder_path, line[2], negative_id)
                negative_id += 1
            else:
                positive_list_file.write('{}.avi\n'.format(positive_id))
                output_path = '{}/{}/{}.avi'.format(output_folder_path, line[2], positive_id)
                positive_id += 1

            out = cv.VideoWriter(output_path, fourcc, 20.0, (640, 480))
            cap.set(cv.CAP_PROP_POS_FRAMES, start)

            # BUG: last frame of the last video may be empty. need to fix ASAP
            for i in range(sequence_length + 1):
                _, frame = cap.read()
                out.write(frame)

            out.release()
