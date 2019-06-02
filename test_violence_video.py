#! /usr/bin/env python
# ffmpeg -i test.avi -vf scale=320:240 test1.avi
# to resize videos to 240 rows and 320 coloums

import math
import sys

import cv2 as cv
import numpy as np
from keras.models import model_from_json

from core import PreProcess, VioFlow

if len(sys.argv) < 2:
    print 'usage: {} VIDEO_PATH MODEL_PATH'.format(sys.argv[0])
    exit()
video_name = sys.argv[1]
model_path = sys.argv[2]

json_file = open('{}/model_100.json'.format(model_path), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("{}/model_100.h5".format(model_path))

video = PreProcess()
video.read_video(video_name)
extractor = VioFlow()

pred = 0
video.current_frame_id = 0
while (True):
    vif = extractor.getFeatureVector(video.get_next_sequence())
    vif = np.reshape(vif, (-1, vif.shape[0]))
    pred = model.predict(vif)
    print pred

    display = video.get_original_frame_from_index(video.current_frame_id)
    if pred > 0.9:
        circle_color = (0, 0, 255)
    else:
        circle_color = (255, 0, 0)
    cv.circle(display, (30, 30), 10, circle_color, -1)
    cv.imshow("display", display)
    cv.waitKey(1)
