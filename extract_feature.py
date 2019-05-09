import sys
import os

from core import VioFlow, PreProcess

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid argument!")
        print("Usage: {} INPUT_FOLDER OUTPUT_FOLDER".format(sys.argv[0]))
        exit()
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    output_folder_positive = output_folder + '/positive/'
    output_folder_negative = output_folder + '/negative/'
    video = PreProcess()
    extractor = VioFlow()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder_positive):
        os.makedirs(output_folder_positive)

    if not os.path.exists(output_folder_negative):
        os.makedirs(output_folder_negative)

    positive_list = open('{}/positive/list.txt'.format(input_folder))
    for each_file in positive_list.readlines():
        each_file = each_file[:-1]
        video.read_video('{}/positive/'.format(input_folder) + each_file)
        out_file = each_file[:-3] + 'txt'
        print each_file + '-----------------------------------------------------'
        extractor.writeFeatureToFile(video.get_next_sequence(), output_folder_positive + out_file)
        print each_file + '  done'

    negative_list = open('{}/negative/list.txt'.format(input_folder))
    for each_file in negative_list.readlines():
        each_file = each_file[:-1]
        video.read_video('{}/negative/'.format(input_folder) + each_file)
        out_file = each_file[:-3] + 'txt'
        print each_file + '-----------------------------------------------------'
        extractor.writeFeatureToFile(video.get_next_sequence(), output_folder_negative + out_file)
        print each_file + '  done'
