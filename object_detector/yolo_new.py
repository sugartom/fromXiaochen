# by XC
# yolo2 object detector 
# processing images: python yolo.py --img_folder [img_folder_path] --read_step 30
# processing video: python yolo.py --video_src [video_path]


# from darkflow.net.build import TFNet
import cv2
import sys 
from time import time, sleep
import argparse
import json 
import os 
from os import listdir
from os.path import isfile, join

# Yitao =================================================================
import grpc
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import olympian_master_grpc_pb2

import numpy as np

from tensorflow.python.framework import tensor_util

# from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def resize_input(data):
    data = cv2.resize(data, (608, 608))
    data = data / 255.
    data = np.expand_dims(data, 0)
    return data

# def process_box(b, h, w, threshold, meta):
#   max_indx = np.argmax(b.probs)
#   max_prob = b.probs[max_indx]
#   label = meta['labels'][max_indx]
#   if max_prob > threshold:
#     left  = int ((b.x - b.w/2.) * w)
#     right = int ((b.x + b.w/2.) * w)
#     top   = int ((b.y - b.h/2.) * h)
#     bot   = int ((b.y + b.h/2.) * h)
#     if left  < 0    :  left = 0
#     if right > w - 1: right = w - 1
#     if top   < 0    :   top = 0
#     if bot   > h - 1:   bot = h - 1
#     mess = '{}'.format(label)
#     return (left, right, top, bot, mess, max_indx, max_prob)
#   return None
# Yitao =================================================================

DEBUG = False

MAX_MESSAGE_LENGTH = 1024 * 1024 * 8

class YOLO:
    def __init__(self, args):
        # opt = { "config": args.darkflow_config,  
        #         "model": args.yolo_model, 
        #         "load": args.yolo_weights, 
        #         "gpuName": args.gpu_id,
        #         "gpu": args.gpu_util,
        #         "threshold": args.yolo_thres}

        # self.tfnet = TFNet(opt)
        self.mode = 'VIDEO'

        self.colors = [ (127,255,127), (255,127,127), (127,127,255), (64,127,255), 
                    (150,255,200), (200,150,255), (255,255,0), (0,255,255), (255,0,255)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Yitao-TLS-Begin
        tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
        tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
        FLAGS = tf.app.flags.FLAGS

        # host, port = FLAGS.server.split(':')
        # channel = implementations.insecure_channel(host, int(port))
        # self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        # channel = grpc.insecure_channel("localhost:50051")
        # self.stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

        channel = grpc.insecure_channel('localhost:50051', options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), 
                                                                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                                                                    ('grpc.max_message_length', MAX_MESSAGE_LENGTH)])
        self.stub = olympian_master_grpc_pb2.OlympianMasterStub(channel)

        self.threshold = args.yolo_thres

        self.labels = open("/home/yitao/Documents/fun-project/devicehive-video-analysis/data/yolo2/yolo2.names").read().splitlines()
        # Yitao-TLS-End

    def get_fname(self, p):
        if '/' in p:
            if p[-1] == '/':
                p = p[:-1]
            p = p.split('/')[-1]
        if self.mode == 'IMG':
            return p

        ps = p.split('.')[:-1]
        res = ''.join(ps)
        print(res)
        return res


    def parse_det(self, dets):
        people = []     
        obj = []   
        for d in dets:
            temp = []
            temp.append(d['topleft']['x'])
            temp.append(d['topleft']['y'])
            temp.append(d['bottomright']['x'])
            temp.append(d['bottomright']['y'])
            if self.mode == 'VIDEO':
                temp.append(d['confidence'])
            temp.append(d['label'])
            
            if temp[-1] == 'person':
                people.append(temp)
            else:
                obj.append(temp)
        
        return people, obj 


    def process_video(self, src, max_frame, visualize):
        self.mode = 'VIDEO'

        cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
        frame_id = 0

        cur_time = time()
        res = {}
        while(True):
            ret, frame = cap.read()
            
            if not ret:
                break 

            frame_id += 1
            if frame_id > max_frame and max_frame > 0:
                break

            if frame_id % 10 == 0:
                print('Frame %d, FPS: %0.1f' % (frame_id, 10/(time() - cur_time)))
                cur_time = time()

            det = self.detect_frame(frame)
            det_people, det_obj = self.parse_det(det)
            det = det_people + det_obj
            res[frame_id] = det

            if visualize:
                frame = self.draw_frame(frame, det)
                cv2.imshow('frame',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return res


    def process_images(self, folder, read_step, name_format, visualize=False):
        self.mode = 'IMG'

        print('Reading image list...')
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        files.sort()
        print('Total images: %d' % len(files))

        res = {}
        img_cnt = 0
        for f in files:
            if img_cnt % read_step != 0:
                img_cnt += 1
                continue 

            print('processing %s..' % f)
            frame = cv2.imread(join(folder, f))
            det = self.detect_frame(frame)
            det_people, det_obj = self.parse_det(det)
            det = det_people + det_obj

            if visualize:
                frame = self.draw_frame(frame, det)
                cv2.imshow('frame',frame)
                cv2.waitKey(0)

            res[f] = det
            img_cnt += 1
        return res 


    def draw_frame(self, frame, dets):
        i = 0
        h, w, _ = frame.shape
        for d in dets:
            left = d[0]
            top = d[1]
            right = d[2]
            bottom = d[3]
            name = d[-1]

            cv2.rectangle(frame, (left, top), (right, bottom),
                            self.colors[i % len(self.colors)], thickness=2)

            cv2.putText(frame, name, (left, top - 15), self.font, fontScale=0.7,
                            color=self.colors[i % len(self.colors)], thickness=2)
            i += 1

        return frame 


    def detect_frame(self, img):
        # print("[Yitao] in yolo.py:detect_frame() is called...!")
        # return self.tfnet.return_predict(img)   

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'devicehive_yolo'
        request.model_spec.signature_name = 'predict_images'

        img = resize_input(img)
        # print(img.shape)

        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(img, dtype = np.float32, shape=img.shape))

        result = self.stub.Predict(request, 10.0)  # 10 secs timeout

        boxes = tensor_util.MakeNdarray(result.outputs['boxes'])
        classes = tensor_util.MakeNdarray(result.outputs['classes'])
        scores = tensor_util.MakeNdarray(result.outputs['scores'])

        boxesInfo = list()
        for i in range(len(boxes)):
            myBox = boxes[i]
            myClass = classes[i]
            myScore = scores[i]

            boxesInfo.append({
                "label": self.labels[myClass],
                "confidence": myScore,
                "topleft": {
                  "x": myBox[1],
                  "y": myBox[0]},
                "bottomright": {
                  "x": myBox[3],
                  "y": myBox[2]}
            })

        return boxesInfo

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_src', dest='video_src')
    parser.add_argument('--max_frame', type=int, dest='max_frame', default=-1)

    parser.add_argument('--img_folder', dest='img_folder')   
    parser.add_argument('--read_step', type=int, dest='read_step', default=1)
    parser.add_argument('--name_format', dest='name_format', default='Left_%d.jpg')
    
    parser.add_argument('--darkflow_config', dest='darkflow_config', 
                            default='/home/smc/darkflow/cfg')
    parser.add_argument('--yolo_model', dest='yolo_model', 
                            default='/home/smc/darkflow/cfg/yolo.cfg')
    parser.add_argument('--yolo_weights', dest='yolo_weights', 
                            default='/home/smc/darknet/yolo.weights')

    parser.add_argument('--yolo_thres', dest='yolo_thres', type=float, default=0.5)
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0)
    parser.add_argument('--gpu_util', dest='gpu_util', type=float, default=0.5)
    
    parser.add_argument('--visualize', dest='visualize', type=bool, default=False)
    parser.add_argument('--json', dest='json', type=bool, default=False)

    return parser.parse_args()


if DEBUG:
    args = parse_input()

    yolo = YOLO(args)

    # process video
    # res = yolo.process_video(args.video_src, args.max_frame, args.visualize)
    
    # processing images 
    res = yolo.process_images(args.img_folder, args.read_step, args.name_format, args.visualize)

    if args.json:
        print('saving to json...')
        json.dump(res, open(yolo.get_fname(args.img_folder) + '.json', 'w'))  

    print('done')
