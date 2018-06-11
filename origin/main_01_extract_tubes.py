import numpy as np
import os
import shutil
import tensorflow as tf
import argparse
import json
import cv2
from tqdm import tqdm
from time import time 

from object_detector import object_detect_batch

# Better
OBJ_DETECT_GRAPH_PATH = './object_detector/zoo/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/out_graph/frozen_inference_graph.pb'
# Faster
# OBJ_DETECT_GRAPH_PATH = './object_detector/zoo/ssd_inception_v2_coco_11_06_2017/out_graph/frozen_inference_graph.pb'

TRACKING_INTERVAL_T = 1.0 # every T seconds run object detector (can be float)
DETECTION_TH = 0.30 # detection threshold for object detector
NO_DETECTION_NEEDED = 2 # any tubelet with detections less than this number will be filtered
NO_FRAMES_TO_KILL_TRACKER = 90 # any tracker that is lost for more than this # of frames will not be searched
MAX_COLOR_HIST_DISTANCE = 0.50 # if the distance is greater than this value, dont match

# TODO experiment with different trackers
# TRACKER_FCN = cv2.TrackerKCF_create # faster
TRACKER_FCN = cv2.TrackerMIL_create # better
# TRACKER_FCN = cv2.TrackerMedianFlow_create # bad

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_video', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=str, required=True)
    parser.add_argument('-v', '--visualize_video', type=int, required=False, default=1)
    parser.add_argument('-s', '--save_tubes', type=int, required=False, default=0)

    args = parser.parse_args()

    gpu = args.gpu
    video_path = args.input_video
    visualize_flag = bool(args.visualize_video)
    save_tubes_flag = bool(args.save_tubes)

    extract_tubes(gpu, video_path, visualize_flag, save_tubes_flag)


def extract_tubes(gpu, video_path, visualize_flag, save_tubes_flag):
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-i', '--input_video', type=str, required=True)
    # parser.add_argument('-g', '--gpu', type=str, required=True)
    # parser.add_argument('-v', '--visualize_video', type=int, required=False, default=1)
    # parser.add_argument('-s', '--save_tubes', type=int, required=False, default=0)

    # args = parser.parse_args()

    print('\nRunning tracking and detection!\n')

    # gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # video_path = args.input_video
    vcap = cv2.VideoCapture(video_path)

    video_name = video_path.split('/')[-1].split('.')[0]

    # visualize_flag = bool(args.visualize_video)
    print('Visualization: %s' % ('yes' if visualize_flag else 'no'))
    # save_tubes_flag = bool(args.save_tubes)
    print('Saving tubes: %s' % ('yes' if save_tubes_flag else 'no'))

    # Video Properties
    vidfps = vcap.get(cv2.CAP_PROP_FPS)
    # sometimes opencv fails at getting the fps
    if vidfps == 0: vidfps = 30

    W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    no_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Object Detector
    detection_graph = object_detect_batch.generate_graph(OBJ_DETECT_GRAPH_PATH)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(graph=detection_graph, config=config)

    # Trackers
    trackers = [] # keeps live trackers
    lost_trackers = [] # keeps the trackers that lost their target
    T_frames = int(vidfps * TRACKING_INTERVAL_T) # every #T_frames run object detector

    frame_no = 0
    tube_id = [0] # its a list so we can change it inplace in the combine_det. function
    
    
    if not visualize_flag: pbar = tqdm(total=no_frames, ncols=100)
    cur_time = time()
    cur_fid = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret: 
            break
        cur_fid += 1
        # resize for speed (debugging)
        # frame = cv2.resize(frame,None,fx=0.5, fy=0.5)

        if frame_no % T_frames == 0:
            # Update the trackers first so that the tracker boxes align better with new detections
            update_trackers(frame, trackers, lost_trackers, frame_no)
            # Get the detection results and combine with tracking
            obj_results = get_detection_results(frame, detection_graph, sess)
            object_detections = process_detection_results(obj_results, H, W)
            combine_detection_with_tracking(frame, frame_no, object_detections, trackers, lost_trackers, tube_id)

            # lost_trackers = []
        else:
            # update trackers with the new frame
            update_trackers(frame, trackers, lost_trackers, frame_no)

        if visualize_flag:
        # if visualize_flag and frame_no > 180: # skipping frames
            imgcv = visualize(frame, trackers, lost_trackers)
            cv2.imshow('Detections - press q to exit', imgcv)
            k = cv2.waitKey(0)

            if k == ord('q'):
                break

        frame_no += 1
        if not visualize_flag: pbar.update(1)
    
    print('Det+Track: %0.2f' % ((time() - cur_time) / cur_fid))
    vcap.release()
    pbar.close()
    sess.close()

    # video is done if there are alive trackers mark their end as last frame
    for tracker in trackers:
        tracker['lost_since_frame'] = frame_no

    if save_tubes_flag:
        # we are interested in tubelets which lasted at least #NO_DETECTION_NEEDED detections
        all_trackers = trackers + lost_trackers
        # trackers_to_use = all_trackers # save everything
        trackers_to_use = [tubelet for tubelet in all_trackers\
                            if len(tubelet['detections']) >= NO_DETECTION_NEEDED]

        process_tubelets(trackers_to_use)

        json_file_path = './jsons/%s_tubes.json' % video_name

        print('\nSaving detected tube information to %s\n' % json_file_path)
        with open(json_file_path, 'w') as fp:
            json.dump(trackers_to_use, fp)
    
        extract_and_save_tubes(video_path, video_name, trackers_to_use)

def extract_and_save_tubes(video_path, video_name, trackers_to_use):
    # make sure to use the trackers that are updated by process_tubelets
    # otherwise opencv will complain about the sizes

    # go thorugh the frames again and extract tubes
    vcap = cv2.VideoCapture(video_path)
    # Video Properties
    fps = vcap.get(cv2.CAP_PROP_FPS)
    no_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    tubes_folder = './tubes'
    vid_tubes_folder = os.path.join(tubes_folder, video_name)
    print('Extracting and saving tubes to %s/' % vid_tubes_folder)
    print('Total of %i tubes detected' % len(trackers_to_use))
    
    if not os.path.exists(vid_tubes_folder): 
        os.mkdir(vid_tubes_folder)
    else:
        print('\t Folder %s exists, deleting...' % vid_tubes_folder)
        shutil.rmtree(vid_tubes_folder)
        os.mkdir(vid_tubes_folder)

    # sort trackers by starting frame
    trackers_to_use.sort(key=lambda x: x['starting_frame'])

    vid_writers = {} #each tubelet needs its own video writer
    frame_cnt = 0
    cur_fid = 0
    pbar = tqdm(total=no_frames, ncols=100)
    while vcap.isOpened():
        ret, frame = vcap.read()
        cur_fid += 1
        if not ret: 
            break

        cur_time = time()
        for tt, tracker in enumerate(trackers_to_use):
            # if tt in completed_indices:
            #     continue
            label = tracker['detections'][0]['class_str']
            tube_id = tracker['tube_id']
            tube_key = '%s_%.3i' % (label, tube_id)
            vid_file_name = '%s.avi' % tube_key

            if frame_cnt < tracker['starting_frame']: 
                break
            elif frame_cnt == tracker['starting_frame']:
                # initialize
                file_path = os.path.join(vid_tubes_folder, vid_file_name)
                box_height = tracker['avg_height']
                box_width = tracker['avg_width']
                # import pdb;pdb.set_trace()
                writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (box_width,box_height))
                vid_writers[tube_key] = writer
                # write first frame
                idx = frame_cnt - tracker['starting_frame'] # 0 in this case
                box = tracker['avg_box_list'][idx]
                extracted_box = extract_box_frame(frame, box)
                # cv2.imshow('test', extracted_box)
                # cv2.waitKey(0)
                vid_writers[tube_key].write(extracted_box)

                # os.mkdir(file_path+'_imgs')
                # cv2.imwrite(file_path+'_imgs/%.3i.jpg' % idx, extracted_box)

            elif frame_cnt < tracker['starting_frame'] + len(tracker['avg_box_list']):
                # update
                idx = frame_cnt - tracker['starting_frame'] # 0 in this case
                box = tracker['avg_box_list'][idx]
                extracted_box = extract_box_frame(frame, box)
                vid_writers[tube_key].write(extracted_box)

                # file_path = os.path.join(tubelet_videos_folder, vid_file_name)
                # cv2.imwrite(file_path+'_imgs/%.3i.jpg' % idx, extracted_box)
                
            elif frame_cnt == tracker['starting_frame'] + len(tracker['avg_box_list']):
                vid_writers[tube_key].release()
                # print('Tube %s completed!' % tube_key)

            else:
                continue
            
        pbar.update(1)
        frame_cnt += 1

    print('To Tube: %0.2f' % ((time() - cur_time)/cur_fid))
    vcap.release()
    pbar.close()



def get_detection_results(frame, detection_graph, sess):
    frame_exp = np.expand_dims(frame, axis=0)
    obj_results = \
            object_detect_batch.get_detections(detection_graph, sess, frame_exp)

    return obj_results

def process_detection_results(res_list, H, W):
    boxes_batch,scores_batch,classes_batch,num_detections_batch = res_list
    mini_batch_size = len(boxes_batch)

    object_detections_batch = []
    for bb in range(mini_batch_size):
        boxes,scores,classes,num_detections = boxes_batch[bb],scores_batch[bb],classes_batch[bb],num_detections_batch[bb]
        cur_object_detections = []
        for ii in range(len(boxes)):
            score = get_3_decimal_float(scores[ii])
            # Filter detections
            if score < DETECTION_TH:
                continue
            d_box = boxes[ii]
            top, left, bottom, right = d_box
            box = (int(top * H), int(left * W), int(bottom * H), int(right * W))
            # box = [get_3_decimal_float(coord) for coord in d_box]

            class_no = int(classes[ii])
            class_str = object_detect_batch.get_object_name(class_no)

            detection = {'box':box, 'score':score, 'class_no':class_no, 'class_str':class_str}

            cur_object_detections.append(detection)

        object_detections_batch.append(cur_object_detections)
    return object_detections_batch  

def update_trackers(frame, trackers, lost_trackers, frame_no):
    ''' each tracker is:
     new_tracker_dict = {'detections': [detection0, detection1], 
                         'tracker':tracker object, 
                         'box':box ([top, left, bottom, right]), 
                         'box_list':[box0, box1, ....],
                         'tube_id':tube_no,
                         'starting_frame' = frame_no}

    '''
    new_trackers = []
    for tracker_info in trackers:
        tracker = tracker_info['tracker']
        try:
            returned, bbox = tracker.update(frame)
        except KeyboardInterrupt:
            raise
        except:
            # sometimes update gives errors if the box slided completely outside the img frame
            # very rarely.
            returned = False
        #    import pdb;pdb.set_trace()
        # tracker bbox is (left, top, width, height)
        if not returned:
            tracker_info['lost_since_frame'] = frame_no
            lost_trackers.append(tracker_info)
        else:
            left, top = bbox[0], bbox[1]
            right, bottom = left + bbox[2], top + bbox[3]
            detection_style_box = [top, left, bottom, right]
            detection_style_box = [int(coord) for coord in detection_style_box]
            tracker_info['box'] = detection_style_box
            tracker_info['box_list'].append(detection_style_box)
            new_trackers.append(tracker_info)

    trackers[:] = new_trackers

def combine_detection_with_tracking(frame, frame_no, obj_results, trackers, lost_trackers, tube_id):
    '''
    If trackers are empty: create new trackers from the new detections
    Else: Calculate all IoUs between detections and trackers = IoU_mtx, if detection and tracker object class dont match IoU = -1.0
        If a detection has a IoU < new_det_IoU_th then create new detections
        Check detections with max IoUs for each tracker
            if two or more trackers match with the same detection box: 
                assign detection to the tracker with maximum detection score that was previously saved (either starting score or updated score)
                other trackers are lost, assign IoU = -2.0
            if max_IoU < max_IoU_for_each_tracklet: that tracker is lost. Check this while updating other trackers

        Check if any of the new detections match with any lost trackers by comparing color histograms in the close vicinity
            if a tracker is lost more than # of frames, dont match it with new ones anymore
                
    '''
    # calculate IoUs
    # detections = obj_results['detections']
    detections = obj_results[0]
    no_detections = len(detections)
    no_trackers = len(trackers)

    H, W, C = frame.shape

    new_trackers = []

    if trackers: # if we are already tracking objects
        if detections:
            IoU_mtx = np.zeros((no_detections, no_trackers))

            # calculate IoUs for each detection for each tracklet
            for dd in range(no_detections):
                for tt in range(no_trackers):
                    cur_detection = detections[dd]
                    detection_box = cur_detection['box']

                    # top, left, bottom, right = detection_box
                    # detection_box = (int(top * H), int(left * W), int(bottom * H), int(right * W))

                    cur_tracker = trackers[tt]
                    tracker_box = cur_tracker['box']

                    # check if they are same class
                    if cur_detection['class_no'] == cur_tracker['detections'][-1]['class_no']:
                        cur_IoU = IoU_box(detection_box, tracker_box)
                    else:
                        cur_IoU = -1.0
                    IoU_mtx[dd, tt] = cur_IoU
        else:
            IoU_mtx = np.ones((1,no_trackers)) * -3.0

        max_IoU_for_each_detection = np.max(IoU_mtx, axis=1)

        #### if a detection has IoU<th with every tracklets, it is a new object
        new_det_IoU_th = 0.50

        # try:
        new_detections = [detections[ii] for ii in range(no_detections) \
                            if max_IoU_for_each_detection[ii] < new_det_IoU_th]
        # except IndexError:
            # import pdb;pdb.set_trace()
            # pass

        #### update the tracklets using aligning detections
        max_IoU_for_each_tracklet = np.max(IoU_mtx, axis=0)

        indices = np.argmax(IoU_mtx, axis=0)

        # find merged tracklets
        repcnt = np.bincount(indices, minlength=no_detections)
        repeated_det_indices = [ii for ii in range(no_detections) if repcnt[ii] > 1]

        # import pdb;pdb.set_trace()

        # if two or more trackers match with one detection
        # look for their confidence score for their last detection
        # and 1- choose the one with most confidence
        # or  2- most similar color histogram
        for merged_index in repeated_det_indices:
            tracker_indices = [ii for ii in range(no_trackers) if indices[ii] == merged_index]
            ### 1- get the initial detection confidence for each tracker and find max
            confidences = [trackers[tracker_index]['detections'][-1]['score'] for tracker_index in tracker_indices] 
            max_conf_indx = tracker_indices[np.argmax(confidences)]
            trackers_to_remove = set(tracker_indices) - set([max_conf_indx])
            ### 2- get the color histograms and find min difference
            # histograms = [trackers[tracker_index]['histogram'] for tracker_index in tracker_indices]
            # cur_detection_box = detections[merged_index]['box']
            # cur_detection_histogram = get_object_color_histogram(frame, cur_detection_box)
            # hist_distances = [cv2.compareHist(tracker_hist, cur_detection_histogram, cv2.HISTCMP_BHATTACHARYYA) \
            #                         for tracker_hist in histograms]
            # min_dist_indx = tracker_indices[np.argmin(hist_distances)]
            # trackers_to_remove = set(tracker_indices) - set([min_dist_indx])


            

            for tracker_index in trackers_to_remove:
                max_IoU_for_each_tracklet[tracker_index] = -2.0


        # if the max_IoU is less then this th, then this tracker is lost
        non_alignment_th = new_det_IoU_th
        for tt in range(no_trackers):
            cur_max_IoU = max_IoU_for_each_tracklet[tt]
            detection_index = indices[tt]
            cur_tracker = trackers[tt]

            if cur_max_IoU < non_alignment_th:
                # import pdb;pdb.set_trace()
                cur_tracker['lost_since_frame'] = frame_no
                lost_trackers.append(cur_tracker)
                continue

            # update the tracker to track the new bounding box
            detection_res = detections[detection_index]
            detection_box = detection_res['box']

            bbox = convert_detection_box_to_tracker_bbox(detection_box)
            
            tracker_obj = TRACKER_FCN()
            tracker_obj.init(frame,bbox)
            
            cur_tracker['tracker'] = tracker_obj
            cur_tracker['box'] = detection_box
            # cur_tracker['box_list'].append(detection_box)
            # since we have updated the box_list with update_trackers in this iteration
            # we have to replace the last box instead of appending
            cur_tracker['box_list'][-1] = detection_box
            cur_tracker['detections'].append(detection_res)

            # update the histogram with the new detection box
            cur_tracker['histogram'] = get_object_color_histogram(frame, detection_box)

            new_trackers.append(cur_tracker)

    else: # trackers list is empty
        new_detections = detections

    match_lost_trackers(frame, lost_trackers, new_detections, new_trackers, frame_no)
    # import pdb;pdb.set_trace()
    #### start tracking the new detections
    for detection in new_detections:
        # ignore if it is not a person
        if detection['class_str'] != 'person': 
            continue
        detection_box = detection['box']

        # top, left, bottom, right = detection_box
        # detection_box = (int(top * H), int(left * W), int(bottom * H), int(right * W))

        ### opencv version 3.2.0 seems buggy. tracking results are off at some cases
        # if cv2.__version__.split('.')[1] == '3':
        #     tracker = TRACKER_FCN() # opencv version >= 3.3
        # else:
        #     tracker = cv2.Tracker_create('KCF') # 3.0 < opencv version < 3.3
        tracker = TRACKER_FCN()

        bbox = convert_detection_box_to_tracker_bbox(detection_box)
        tracker.init(frame,bbox)

        new_tracker_info = {}
        new_tracker_info['tracker'] = tracker
        new_tracker_info['box'] = detection_box
        new_tracker_info['box_list'] = [detection_box]
        new_tracker_info['detections'] = [detection]
        tube_id[0] += 1
        new_tracker_info['tube_id'] = tube_id[0]
        new_tracker_info['starting_frame'] = frame_no

        # add the histogram of the detection
        new_tracker_info['histogram'] = get_object_color_histogram(frame, detection_box)
        new_trackers.append(new_tracker_info)
        # import pdb;pdb.set_trace()
        # print('creating new')


    trackers[:] = new_trackers


def match_lost_trackers(frame, lost_trackers, new_detections, new_trackers, frame_no):
    # check histograms of new detection and try to match with lost trackers
    hist_mtx = None
    H,W,C = frame.shape
    
    updated_lost_trackers = []
    if lost_trackers and new_detections:
        hist_mtx = np.zeros([len(lost_trackers), len(new_detections)])

        # calculate hellinger(HISTCMP_BHATTACHARYYA) distance between every lost tracker and new detection
        for tt, lost_tracker in enumerate(lost_trackers):
            tracker_hist = lost_tracker['histogram']
            tracker_box = lost_tracker['box_list'][-1]
            tracker_ctr = (float(tracker_box[2] + tracker_box[0])/2.0, float(tracker_box[3] + tracker_box[1])//2.0)

            for dd, new_detection in enumerate(new_detections):
                detection_box = new_detection['box']
                detection_hist = get_object_color_histogram(frame, detection_box)
                detection_ctr = (float(detection_box[2] + detection_box[0])/2.0, float(detection_box[3] + detection_box[1])/2.0)

                ctr_distance = ((tracker_ctr[0] - detection_ctr[0])**2.0 + (tracker_ctr[1] - detection_ctr[1])**2.0)**(0.5)
                if new_detection['class_str'] != lost_tracker['detections'][-1]['class_str']:
                    hist_mtx[tt, dd] = 20.0
                # elif ctr_distance < min(H,W)/4:
                elif ctr_distance < max(tracker_box[2] - tracker_box[0], tracker_box[3] - tracker_box[1]): #
                    hist_mtx[tt, dd] = cv2.compareHist(tracker_hist, detection_hist, cv2.HISTCMP_BHATTACHARYYA)
                    # print(hist_mtx[tt, dd])
                    # cv2.imshow('track', frame[tracker_box[0]:tracker_box[2]+1, tracker_box[1]:tracker_box[3]+1])
                    # cv2.imshow('det', frame[detection_box[0]:detection_box[2]+1, detection_box[1]:detection_box[3]+1])
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('track')
                    # cv2.destroyWindow('det')
                else:
                    hist_mtx[tt, dd] = 10.0
        matched_detection_indices = []

        for tt, cur_tracker in enumerate(lost_trackers):
            min_hist_distance = np.min(hist_mtx[tt,:])
            min_hist_index = np.argmin(hist_mtx[tt,:])
            
            # if the min BHATTACHARYYA distance is less than some threshold, we matched the lost tracker
            # if the trackers was lost for a number of frames, do not refresh it
            if min_hist_distance < MAX_COLOR_HIST_DISTANCE and frame_no - cur_tracker['lost_since_frame'] + 1 < NO_FRAMES_TO_KILL_TRACKER:    
                matched_detection_indices.append(min_hist_index)
                matched_detection = new_detections[min_hist_index]
                detection_box = matched_detection['box']

                tracker_obj = TRACKER_FCN()
                bbox = convert_detection_box_to_tracker_bbox(detection_box)
                tracker_obj.init(frame,bbox)
                
                cur_tracker['tracker'] = tracker_obj
                
                # cur_tracker['box_list'].append(detection_box)
                # interpolate for the frames the tracker was lost
                no_interpolate = frame_no - cur_tracker['lost_since_frame'] + 1
                interpolate_boxes(cur_tracker['box_list'], detection_box, no_interpolate)
                cur_tracker['box'] = detection_box
                cur_tracker['detections'].append(matched_detection)

                # update the histogram with the new detection box
                cur_tracker['histogram'] = get_object_color_histogram(frame, detection_box)

                new_trackers.append(cur_tracker)
            
            # if it is greater than this tracker is still lost
            else:
                updated_lost_trackers.append(cur_tracker)
        
        lost_trackers[:] = updated_lost_trackers
        # delete the matched detections from the new_detections list
        # since they are matched and we dont have to recreate them
        matched_detection_indices = list(set(matched_detection_indices))
        for ii in sorted(matched_detection_indices, reverse=True):
            try:
                del new_detections[ii]
            except IndexError:
                import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
    
def interpolate_boxes(box_list, new_box, no_frames):
    last_box = box_list[-1]
    # l_top, l_left, l_bottom, l_right = last_box
    last_box = np.array(last_box)

    # n_top, n_left, n_bottom, n_right = new_box
    new_box = np.array(new_box)

    slopes = (new_box - last_box) / float(no_frames)

    for ff in range(no_frames):
        cur_box = last_box + (ff+1) * slopes
        cur_box = [int(coord) for coord in cur_box]
        box_list.append(cur_box)

        # imgcv = np.copy(frame)
        # top, left, bottom, right = cur_box
        # cv2.rectangle(imgcv, (left,top), (right,bottom), (0,255,255), 1)
        # cv2.imshow('interp', imgcv)
        # cv2.waitKey(0)
        
    # import pdb;pdb.set_trace()

    # box_list.append(new_box)


def get_object_color_histogram(frame, box):
    top, left, bottom, right = box
    object_bbox = frame[top:bottom+1, left:right+1]
    # cv2.imshow('object_bbox', object_bbox)
    hist = cv2.calcHist([object_bbox], [0, 1, 2], None, [8,8,8], [0, 255, 0, 255, 0, 255])
    cv2.normalize(hist, hist)
    hist = hist.flatten()
    return hist

def IoU_box(box1, box2):
    # detection box is (ymin, xmin, ymax, xmax) or (top, left, bottom, right)
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2

    xA1 = left1
    yA1 = top1
    xA2 = right1
    yA2 = bottom1

    xB1 = left2
    yB1 = top2
    xB2 = right2
    yB2 = bottom2

    xI1 = max(xA1, xB1)
    yI1 = max(yA1, yB1)

    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)

    areaIntersection = max(0, xI2 - xI1) * max(0, yI2 - yI1)

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    
    IoU = areaIntersection / float(areaA + areaB - areaIntersection)
    return IoU

TUBE_COLORS = np.random.rand(300,3) * 255
TUBE_COLORS = TUBE_COLORS.astype(int)

def visualize(frame, trackers, lost_trackers):
    imgcv = np.copy(frame)
    colors = np.load('object_detector/colors.npy')

    for tracker in trackers:
        box = tracker['box']
        detection_info = tracker['detections'][-1]

        top, left, bottom, right = box

        conf = detection_info['score']
        label = detection_info['class_str']

        label_indx = min(detection_info['class_no'] - 1, 79)

        tube_id = tracker['tube_id']

        message = '%.3i_%s: %.2f' % (tube_id, label , conf)

        print(message)

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
        

    print('\n\n')
    return imgcv

def get_3_decimal_float(infloat):
    return float('%.3f' % infloat)

def convert_detection_box_to_tracker_bbox(detection_box):
    # convert detection style box to tracker bbox
    top, left, bottom, right = detection_box
    bHeight = bottom - top
    bWidth = right - left
    bbox = (left, top, bWidth, bHeight)

    return bbox

def process_tubelets(trackers):
    '''
    changes the 'trackers' in place
    Process the tubelets such that frames in each tubelets are consistent
    1 - find the average area and H-W ratio for a tubelet
    2 - find the center point for each frame in a tubelet
    3 - remove the opencv tracker object since it is not saveable
    4 - smmoth the centers using moving average filter
    5 - for every frame save the new averaged size bounding boxes in
        a new field ['averaged_boxes']
    
    '''
    for tubelet in trackers:
        del tubelet['tracker'] # for saving
        del tubelet['histogram'] # for saving

        box_list = tubelet['box_list']

        # calculate the average box
        # find mean area and mean box ratio ( x / y)
        areas = []
        ratios = []
        # edges = []
        edges_x = []
        edges_y = []
        for box in box_list:
            top, left, bottom, right = box

            edge_x = right - left
            edge_y = bottom - top

            area = edge_x * edge_y
            ratio = edge_x / float(edge_y)

            areas.append(area)
            ratios.append(ratio)

            # edges.extend([edge_x, edge_y])
            edges_x.append(edge_x)
            edges_y.append(edge_y)

        box_size = np.mean(areas)
        box_ratio = np.mean(ratio)
        # max_edge = max(edges)
        max_edge_x = max(edges_x)
        max_edge_y = max(edges_y)

        # sqrt = lambda x: x ** (0.5)
        # box_y = sqrt(box_size / box_ratio)
        # box_x = box_y * box_ratio
        box_y = max_edge_y
        box_x = max_edge_x

        box_x = int(box_x)
        box_y = int(box_y)

        # add some scaling
        # box_x = int(box_x * 1.25)
        # box_y = int(box_y * 1.25)

        tubelet['avg_height'] = box_y
        tubelet['avg_width'] = box_x
        # update the boxes using the previous box center and calculated average box
        tubelet['avg_box_list'] = []

        # Smooth the centers using moving average
        N = 3
        cumsum = [(0,0)] 
        # centers = []
        for jj, box in enumerate(box_list):
            top, left, bottom, right = box
            
            y_center = (bottom + top) / 2
            x_center = (right + left) / 2

            ii = jj+1
            # # import pdb; pdb.set_trace()
            # # smoothing
            cumsum.append((cumsum[ii-1][0] + x_center, cumsum[ii-1][1] + y_center))
            if ii >= N:
                moving_avg_x = (cumsum[ii][0] - cumsum[ii-N][0]) // N
                moving_avg_y = (cumsum[ii][1] - cumsum[ii-N][1]) // N

                y_center = moving_avg_y
                x_center = moving_avg_x



            top = y_center - box_y / 2
            left = x_center - box_x / 2

            bottom = y_center + box_y / 2
            right = x_center + box_x / 2

            avg_box = [top, left, bottom, right]
            
            tubelet['avg_box_list'].append(avg_box)

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
