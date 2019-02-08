#!/usr/bin/env python3
import core

if __name__ == '__main__':
    video = core.Video("assets/test.avi")
    vif = video.next_sequence().get_flows().get_vif()
    print(vif)
