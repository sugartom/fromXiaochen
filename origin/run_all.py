import argparse
import os

import main_01_extract_tubes
import main_02_detect_actions
import main_03_visualize_and_generate_video


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_video', type=str, required=True)
    # parser.add_argument('-o', '--object_gpu', type=str, required=True)
    # parser.add_argument('-a', '--action_gpu', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=str, required=True)
    # parser.add_argument('-s', '--save_tubes', type=int, required=False, default=0)

    args = parser.parse_args()

    gpu = args.gpu
    video_path = args.input_video
    video_name = video_path.split('/')[-1].split('.')[0]
    tubes_folder = './tubes'
    vid_tubes_folder = os.path.join(tubes_folder, video_name)

    print('\n\nRunnning everything on the video %s\n\n' % video_path)

    main_01_extract_tubes.extract_tubes(gpu, video_path, visualize_flag=0, save_tubes_flag=1)
    main_02_detect_actions.detect_actions(gpu, vid_tubes_folder, visualize_flag=0)
    main_03_visualize_and_generate_video.visualize_generate_result(video_path, visualize_flag=0, save_video=1)


if __name__ == '__main__':
    main()