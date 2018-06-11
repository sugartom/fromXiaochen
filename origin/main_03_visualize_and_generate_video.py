import numpy as np
import os
import json
import cv2
import argparse
from time import time

VIDEO_OUT_FOLDER = './output_videos/'

TUBE_COLORS = np.random.rand(300,3) * 255
TUBE_COLORS = TUBE_COLORS.astype(int)


with open('label_conversion.json') as fp:
    LABEL_CONV = json.load(fp)
    # "training2real": {"0": [15, "answer_phone"], ...
    # "real2training": {"15": [0, "answer_phone"], ...


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_video', type=str, required=True)
    parser.add_argument('-v', '--visualize_video', type=int, required=False, default=1)
    parser.add_argument('-s', '--save_video', type=int, required=False, default=1)

    args = parser.parse_args()

    video_path = args.input_video
    visualize_flag = bool(args.visualize_video)
    save_video = bool(args.save_video)

    visualize_generate_result(video_path, visualize_flag, save_video)
    

def visualize_generate_result(video_path, visualize_flag, save_video):

    if save_video: print('\nGenerating output video!\n')

    video_name = video_path.split('/')[-1].split('.')[0]

    print('Visualization: %s' % ('yes' if visualize_flag else 'no'))
    print('Saving output video: %s' % ('yes' if save_video else 'no'))

    tubes_json = './jsons/%s_tubes.json' % video_name
    actions_json = './jsons/%s_actions.json' % video_name

    if os.path.exists(tubes_json):
        with open(tubes_json) as fp:
            tubelet_infos = json.load(fp)
    else:
        print('Error: Tubes json not found! Run extract tubes first')
        raise IOError

    if os.path.exists(actions_json):
        with open(actions_json) as fp:
            actions_info = json.load(fp)
    else:
        print('Error: Actions json not found! Run detect actions first')
        raise IOError

    
    tubelet_ids = actions_info.keys()
    
    # combine annotations
    tubelets_w_actions = []
    for tubelet_info in tubelet_infos:
        tubelet_id = tubelet_info['tube_id']
        tubelet_key = 'person_%.3i' % tubelet_id
        action_probs = actions_info[tubelet_key]
        tubelet_length = len(tubelet_info['avg_box_list'])
        action_length = len(action_probs)
        slope = action_length / float(tubelet_length)

        # if tubelet_id == 3: import pdb;pdb.set_trace()

        # TODO add some filtering/smoothing here
        actions_for_each_frame = []
        for ii in range(tubelet_length):
            cur_action = action_probs[int(ii * slope)]
            actions_for_each_frame.append(cur_action)
        tubelet_info['frame_actions'] = actions_for_each_frame
        tubelet_info['tube_key'] = tubelet_key

    vcap = cv2.VideoCapture(video_path)

    ## Video Properties
    vidfps = vcap.get(cv2.CAP_PROP_FPS)
    # sometimes opencv fails at getting the fps
    if vidfps == 0: vidfps = 20

    W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    H, W = int(H), int(W)

    if save_video:
        out_path = VIDEO_OUT_FOLDER + '%s_results.avi' % video_name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_path,fourcc, vidfps, (5*W//4,H))

    frame_no = 0
    cur_time = time()
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret: break

        trackers = []
        lost_trackers = []
        for tubelet_info in tubelet_infos:
            start_frame = tubelet_info['starting_frame']
            end_frame = tubelet_info['lost_since_frame']
            if frame_no >= start_frame and frame_no < end_frame:
                relative_frame = frame_no - start_frame
                current_box = tubelet_info['avg_box_list'][relative_frame]
                current_detections = [tubelet_info['detections'][0]]
                tubelet_id = tubelet_info['tube_id']
                tube_key = tubelet_info['tube_key']
                current_actions = tubelet_info['frame_actions'][relative_frame]
                new_tracker = {'box': current_box,
                               'detections': current_detections,
                               'tube_id': tubelet_id,
                               'tube_key': tube_key,
                               'action_probs': current_actions}
                trackers.append(new_tracker)

            elif frame_no > start_frame and frame_no <= end_frame + 30:
                # relative_frame = frame_no - start_frame
                current_box = tubelet_info['avg_box_list'][-1]
                current_detections = [tubelet_info['detections'][0]]
                tubelet_id = tubelet_info['tube_id']
                current_actions = tubelet_info['frame_actions'][-1]
                new_tracker = {'box': current_box,
                               'detections': current_detections,
                               'tube_id': tubelet_id,
                               'tube_key': tube_key,
                               'action_probs': current_actions}
                lost_trackers.append(new_tracker)
            else:
                continue
        
        img_with_objects = visualize(frame, trackers, lost_trackers, visualize_flag)
        sidebar = add_action_sidebars(frame, trackers, H, W)

        img_with_actions = np.concatenate([sidebar, img_with_objects], axis=1)

        if visualize_flag:
            cv2.imshow('Results', img_with_actions)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
        if save_video:
            out.write(img_with_actions)

        frame_no += 1

    print('To result: %0.2f' % ((time() - cur_time)/frame_no))
    vcap.release()
    if save_video: 
        out.release()
        print('Video %s is saved!' % out_path)




def visualize(frame, trackers, lost_trackers, visualize_flag):
    imgcv = np.copy(frame)
    colors = np.load('object_detector/colors.npy')

    for tracker in trackers:
        box = tracker['box']
        detection_info = tracker['detections'][-1]

        top, left, bottom, right = box

        # conf = detection_info['score']
        label = detection_info['class_str']

        label_indx = min(detection_info['class_no'] - 1, 79)

        tube_id = tracker['tube_id']

        # message = '%.3i_%s: %.2f' % (tube_id, label , conf)
        message = '%.3i_%s' % (tube_id, label)

        if visualize_flag: print(message)

        thick = 3
        label_indx = int(label_indx)
        # color = colors[label_indx]
        color = TUBE_COLORS[tube_id]
        cv2.rectangle(imgcv, (left,top), (right,bottom), color, thick)

        font_size =  max(0.5,(right - left)/50.0/float(len(message)))
        # font_size = (right - left)/float(len(message))/10.0
        cv2.rectangle(imgcv, (left, top-int(font_size*40)), (right,top), color, -1)
        cv2.putText(imgcv, message, (left, top-12), 0, font_size, (255,255,255)-color, thick//2)

    for lost_tracker in lost_trackers:
        box = lost_tracker['box']
        top, left, bottom, right = box
        cv2.rectangle(imgcv, (left,top), (right,bottom), (0,0,255), 1)
        

    if visualize_flag:print('\n\n')
    return imgcv

def add_action_sidebars(frame, trackers, H, W):
    black_bar_left = np.zeros([H,W//4,3], np.uint8)
    # black_bar_right = np.zeros([H,W//4,C], np.uint8)

    if trackers:
        for ii, tracker in enumerate(trackers):
            box = tracker['box']
            top, left, bottom, right = box
            bb_label = tracker['tube_key']
            bb_frame = extract_box_frame(frame, box)
            bb_frame = np.copy(bb_frame)
            bbH, bbW = H//8, int(bb_frame.shape[1] / float(bb_frame.shape[0]) * H // 8)
            bb_frame = cv2.resize(bb_frame, (bbW, bbH))
            current_action_probs = tracker['action_probs']
            bb_acts = get_act_strs(current_action_probs, 5)

            # bbH, bbW, bbC = bb_frame.shape
            starting_index = H//24 + ii * H//8 + ii * 10
            if starting_index + bbH < H:
                black_bar_left[starting_index: starting_index+bbH, 20:20+bbW, :] = bb_frame
                
                cur_HH = starting_index + 10
                # label
                font_size = 0.5
                tube_color = TUBE_COLORS[tracker['tube_id']]
                cv2.putText(black_bar_left, bb_label, (40+bbW, cur_HH), 0, font_size, tube_color, 1)
                # Actions
                for act in bb_acts:
                    cur_HH += 13
                    cv2.putText(black_bar_left, act, (40+bbW, cur_HH+13), 0, font_size, (255,255,255), 1)
    
    
    return black_bar_left
            # try:
            #     black_bar_left[cur_H:cur_H+bbH, 20:20+bbW, :] = bb_frame
            #     # label
            #     cv2.putText(black_bar, bb_label, (40+bbW, cur_H + 20), 0, font_size, (255,255,255), thick//2)
            #     # Actions
            #     cur_HH = cur_H + 30
            #     for act in bb_acts:
            #         cur_HH += 13
            #         cv2.putText(black_bar, act, (40+bbW, cur_HH), 0, font_size, (255,255,255), thick//2)
            # except ValueError:
            #     break


def get_act_strs(action_probs, topk):

    class_probs = np.array(action_probs)
    class_order = np.argsort(class_probs) [::-1]
    probs_order = class_probs[class_order]
    class_strs = [LABEL_CONV['training2real'][str(class_no)][1] for class_no in class_order]

    printable = [ '%s : %.3f' % (class_str, prob) for class_str, prob in zip(class_strs,probs_order)]
    
    return printable[0:topk]
    # print(printable)

    # no_print = 5
    # pixel_distance = 25
    # black_bar = np.zeros([(no_print+2) * pixel_distance, T * W,C], np.uint8)

    # for ii in range(no_print):
        # cur_printable = printable[ii]
        # cv2.putText(black_bar, cur_printable, (T * W // 2, (ii+2) * pixel_distance), 0, 1, (255,255,255), 1)

        
def extract_box_frame(frame, box):
    # extracts the box from the full frame
    # takes into account if the box coords are outside of frame boundaries and fills with zeros
    H, W, C = frame. shape
    top, left, bottom, right = box

    # initialize with zeros so out of boundary areas are black
    extracted_box = np.zeros((bottom - top, right - left, 3), np.uint8)

    # sometimes tracker get confused and gives a box completely outside img boundaries
    if left >= W or top >= H:
        # then just return a black frame
        print('Tracker box completely out of range')
        return extracted_box

    if left >= 0: # bounding box coords are within frame boundary
        frame_left = left
        ebox_left = 0
    else: # bounding box coords are outside frame boundary
        frame_left = 0
        ebox_left = 0 - left

    if top >= 0: # bounding box coords are within frame boundary
        frame_top = top
        ebox_top = 0
    else: # bounding box coords are outside frame boundary
        frame_top = 0
        ebox_top = 0 - top

    if right <= W: # bounding box coords are within frame boundary
        frame_right = right
        ebox_right = extracted_box.shape[1]
    else: # bounding box coords are outside frame boundary
        frame_right = W
        ebox_right = extracted_box.shape[1] + (W - right)

    if bottom <= H: # bounding box coords are within frame boundary
        frame_bottom = bottom
        ebox_bottom = extracted_box.shape[0]
    else: # bounding box coords are outside frame boundary
        frame_bottom = H
        ebox_bottom = extracted_box.shape[0] + (H - bottom)

    extracted_box[ebox_top:ebox_bottom, ebox_left:ebox_right, :] = \
        frame[frame_top:frame_bottom, frame_left:frame_right, :]

    # try:
    #     extracted_box[ebox_top:ebox_bottom, ebox_left:ebox_right, :] = \
    #         frame[frame_top:frame_bottom, frame_left:frame_right, :]
    # except ValueError:
    #     import pdb;pdb.set_trace()
    
    return extracted_box

if __name__ == '__main__':
    main()  