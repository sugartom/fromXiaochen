# by XC
# yolo2 object detector 
# processing images: python yolo.py --img_folder [img_folder_path] --read_step 30
# processing video: python yolo.py --video_src [video_path]


from darkflow.net.build import TFNet
import cv2
import sys 
from time import time, sleep
import argparse
import json 
import os 
from os import listdir
from os.path import isfile, join

DEBUG = False

class YOLO:
    def __init__(self, args):
        opt = { "config": args.darkflow_config,  
                "model": args.yolo_model, 
                "load": args.yolo_weights, 
                "gpuName": args.gpu_id,
                "gpu": args.gpu_util,
                "threshold": args.yolo_thres}

        self.tfnet = TFNet(opt)
        self.mode = 'VIDEO'

        self.colors = [ (127,255,127), (255,127,127), (127,127,255), (64,127,255), 
                    (150,255,200), (200,150,255), (255,255,0), (0,255,255), (255,0,255)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX


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
        print("[Yitao] in yolo.py:detect_frame() is called...!")
        return self.tfnet.return_predict(img)   



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
