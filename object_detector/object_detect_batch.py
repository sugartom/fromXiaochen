import tensorflow as tf
import numpy as np
import cv2
import os
import json

from tqdm import tqdm
import pdb

# import batch_segments

BATCH_SIZE = 128

PATH_TO_AVA = '/home/oytun/Dropbox/Python/AVA'

PATH_TO_OBJ = PATH_TO_AVA + '/object_detection'
PATH_TO_CKPT = PATH_TO_OBJ + '/zoo/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/out_graph/frozen_inference_graph.pb'
# PATH_TO_CKPT = PATH_TO_OBJ + '/zoo/ssd_inception_v2_coco_11_06_2017/out_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = PATH_TO_OBJ + '/models/object_detection/data/mscoco_label_map.pbtxt'

#AVA 
AVA_STORAGE_PATH = '/media/storage_brain/ulutan/AVA'

OUT_JSONS_FOLDER = AVA_STORAGE_PATH + '/RCNN_object_jsons/'
# OUT_JSONS_FOLDER = AVA_STORAGE_PATH + '/object_jsons_test/'

OUT_JSONS_PATH = OUT_JSONS_FOLDER + 'objects.'

this_dir, this_filename = os.path.split(__file__)
CAT_INDEX = np.load(os.path.join(this_dir, 'obj_category_index.npy'))[()] # {1:{'id':1, 'name':'person'}}

def get_2_decimal_float(infloat):
    return float('%.2f' % infloat)

def generate_graph(path_to_ckpt):
    # Read the detection graph from the ckpt
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def get_detections(detection_graph, sess, image_np):
    '''image_np can be a batch or a single image with batch dimension 1, dims:[None, None, None, 3]'''
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        # feed_dict={image_tensor: image_np[0:50,:,:,:]})
        feed_dict={image_tensor: image_np[:,:,:,:]})

    return boxes,scores,classes,num_detections

def get_detections_in_batches(detection_graph, sess, image_np):
    batch_size = 9
    no_samples = image_np.shape[0]

    for ii in range(0,no_samples,batch_size):
        cur_batch = image_np[ii:ii+batch_size,:,:,:]
        c_boxes,c_scores,c_classes,c_num_detections = get_detections(detection_graph, sess, cur_batch)
        if ii == 0:
            boxes = c_boxes; scores = c_scores; classes = c_classes; num_detections = c_num_detections
        else:
            boxes = np.concatenate((boxes, c_boxes), axis=0)
            scores = np.concatenate((scores, c_scores), axis=0)
            classes = np.concatenate((classes, c_classes), axis=0)
            num_detections = np.concatenate((num_detections, c_num_detections), axis=0)
    return boxes, scores, classes, num_detections

def get_object_name(classno):
    return CAT_INDEX[classno]['name']

# # Video Reading
# def frame_generator_video(vid):
#     # vids_list = os.listdir(VID_PATH)
#     # vids_list.sort()
#     # for vid in vids_list:
#     vid_loc = os.path.join(VID_PATH, vid)
#     cap = cv2.VideoCapture(vid_loc)
#     vid_fps = cap.get(cv2.CAP_PROP_FPS)
#     skip = vid_fps // FPS
    
#     frame_idx = 0
#     # while cap.isOpened():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = frame[:,:,::-1] # BGR to RGB
#         if frame_idx % skip == 0 :
#             yield True, [frame, frame_idx]
#         frame_idx += 1

#     cap.release()
#     yield False, []
#     return
        
# def bb2annotations(H,W,box,score,class_no):
#     ymin, xmin, ymax, xmax = box
#     topleft = {'x':int(xmin * W) , 'y':int(ymin * H)}
#     bottomright = {'x':int(xmax * W) , 'y':int(ymax * H)}
#     label = CAT_INDEX[class_no]['name']

#     bb_res = {'topleft':topleft, 
#                 'bottomright':bottomright,
#                 'label':label,
#                 'label_no':int(class_no),
#                 'confidence': get_2_decimal_float(score)}
#     return bb_res


# def process_batch_results(objects_list, res_list, frames, B, H, W):
#     boxes_batch,scores_batch,classes_batch,num_detections_batch = res_list
#     for ii in range(B):
#         frame = frames[ii]
#         frame_dict = {'frame_path': frame, 'frame_no':ii, 'results':[]}

#         for jj in range(boxes_batch.shape[1]):
#             box = boxes_batch[ii,jj]
#             score = scores_batch[ii,jj]
#             class_no = classes_batch[ii,jj]
#             bb_res = bb2annotations(H,W, box, score, class_no)
#             frame_dict['results'].append(bb_res)

#         objects_list.append(frame_dict)
        
# def process_batch_results_with_th(objects_list, res_list, frames, B, H, W, th):
#     boxes_batch,scores_batch,classes_batch,num_detections_batch = res_list
#     for ii in range(B):
#         frame = frames[ii]
#         frame_dict = {'frame_path': frame, 'frame_no':int(frame), 'results':[]}

#         for jj in range(boxes_batch.shape[1]):
#             box = boxes_batch[ii,jj]
#             score = scores_batch[ii,jj]
#             class_no = classes_batch[ii,jj]
#             bb_res = bb2annotations(H,W, box, score, class_no)
#             if score > th:
#                 frame_dict['results'].append(bb_res)

#         objects_list.append(frame_dict)

# def run_ava_videos():

#     detection_graph = generate_graph(PATH_TO_CKPT)

#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True

#     with detection_graph.as_default():
#         with tf.Session(graph=detection_graph, config=config) as sess:
#             gen = batch_segments.seg_provider_async()
            
#             pbar = tqdm(total=51070)
#             for ret, async_batch in batch_segments.epoch_wrapper(gen):
#                 if not ret:break
#                 batch, paths = async_batch.get(10000)
#                 vid_str = paths[0].split('/')
#                 vid_name = vid_str[-3] + '.' + vid_str[-2]

#                 vid_json = {'Objects':[], 'Name':vid_name, 'Split': vid_str[-4], 'Vid_path':paths[0][:-8]}

#                 # res_list = get_detections(detection_graph, sess,batch)
#                 res_list = get_detections_in_batches(detection_graph, sess, batch)
#                 # pdb.set_trace()

#                 B,H,W,C = batch.shape
#                 process_batch_results(vid_json['Objects'], res_list, paths, B, H, W)

#                 json_name = OUT_JSONS_PATH + vid_name + '.json'
#                 # pdb.set_trace()
#                 with open(json_name, 'w') as fobj:
#                   json.dump(vid_json, fobj)
#                   print(json_name + ' is written!')
#                 pbar.update(1)
#             pbar.close()
    


# def add_frame_nos():
#     json_list = os.listdir(OUT_JSONS_FOLDER)
#     for cur_json_file in tqdm(json_list):
#         cur_json_path = os.path.join(OUT_JSONS_FOLDER, cur_json_file)
#         with open(cur_json_path) as fid:
#             vid_json = json.load(fid)

#         for frame_no in range(len(vid_json['Objects'])):
#             vid_json['Objects'][frame_no]['frame_no'] = frame_no

#         # with open(cur_json_path, 'w') as fobj:
#         #     json.dump(vid_json, fobj)
#         #     print(cur_json_path + ' is written!')



# if __name__ == '__main__':
#     # add_frame_nos()
#     run_ava_videos()



