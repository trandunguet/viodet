#!/usr/bin/env python3

import sys

import cv2 as cv

class segment:
    def __init__(self, begin, label):
        self.begin = int(begin)
        self.label = label
        print("start new segment: {}".format(label))

    def end(self, end, output):
        self.end = int(end)
        if self.label == 'none':
            return
        output.write("{} {} {}\n".format(self.begin, self.end, self.label))
        print("write: {} {} {}".format(self.begin, self.end, self.label))
        

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid argument!")
        print("Usage: {} VIDEO_PATH OUTPUT".format(sys.argv[0]))
        exit()

    video_path = sys.argv[1]
    output_path = sys.argv[2]

    output = open(output_path, 'w')

    cap = cv.VideoCapture(video_path)

    right_arrow = 83
    left_arrow = 81
    up_arrow = 82
    down_arrow = 84
    escape_key = 27
    n_key = 110
    space_key = 32
    p_key = 112
    key = right_arrow
    current_segment = None
    
    while True:
        frame_id = cap.get(cv.CAP_PROP_POS_FRAMES)

        if key == left_arrow:
            frame_id -= 2
        elif key == up_arrow:
            frame_id += 30
        elif key == down_arrow:
            frame_id -= 30

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)

        ready, frame = cap.read()
        if not ready:
            frame_id = cap.get(cv.CAP_PROP_FRAME_COUNT)
            if current_segment:
                current_segment.end(frame_id, output)
            break
        
        cv.imshow(video_path, frame)
        key = cv.waitKey()

        if key == n_key:
            label = 'negative'
        elif key == p_key:
            label = 'positive'
        elif (key == space_key) or (key == escape_key):
            label == 'none'
        else:
            continue

        if current_segment:
            current_segment.end(frame_id, output)
        
        current_segment = segment(frame_id, label)
        if key == escape_key:
            break

    output.close()
