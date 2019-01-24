import sys
import cv2 as cv

sys.path.append('./')

from video_process import VideoProcess

if __name__ == '__main__':
    video = VideoProcess("assets/test.avi")
    video.process_next_sequence()
