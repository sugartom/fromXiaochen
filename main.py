from extract_tube import TubeExtractor
from detect_action import ActionDetector
from display import Visualizer
import threading 
import sys 
from time import time, sleep
from queue_manager import Q
import argparse

def parse_input():
	parser = argparse.ArgumentParser()
	# INPUT 
	parser.add_argument('--video_src')
	parser.add_argument('--max_frame', type=int, default=-1)

	parser.add_argument('--video_fps', type=int, default=30)
	parser.add_argument('--batch', type=int, default=1)
	
	# INFRA
	parser.add_argument('--tube_queue_size', type=int, default=10)
	parser.add_argument('--action_queue_size', type=int, default=10)
	parser.add_argument('--frame_queue_size', type=int, default=1000)

	# TUBE_EXTRACTOR
	parser.add_argument('--darkflow_config', default='/home/yitao/Documents/fun-project/darknet-repo/darkflow/cfg')
	parser.add_argument('--yolo_model', default='/home/yitao/Documents/fun-project/darknet-repo/darkflow/cfg/yolo.cfg')
	parser.add_argument('--yolo_weights', default='/home/yitao/Documents/fun-project/darknet-repo/darkflow/bin/yolo.weights')

	parser.add_argument('--yolo_thres', type=float, default=0.25)
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--gpu_util', type=float, default=0.8)

	parser.add_argument('--max_age', type=int, default=30)
	parser.add_argument('--min_hits', type=int, default=3)

	# ACTION DETECTOR
	parser.add_argument('--action_ckpt', default='/home/yitao/Documents/fun-project/actions_demo/action_detector/models/weights/model_ckpt-3')
	parser.add_argument('--min_tube_len', type=int, default=10)
	
	# OUTPUT 
	parser.add_argument('--stat', type=bool, default=True)
	parser.add_argument('--visualize', type=bool, default=False)
	parser.add_argument('--save_vid', type=bool, default=True)
	parser.add_argument('--json', type=bool, default=False)

	return parser.parse_args()


######################## MAIN FUNCTION ########################### 

# read parameters 
args = parse_input()

# global para
running = [True]

# init the infra
tube_queue = Q(args.tube_queue_size) 
action_queue = Q(args.action_queue_size) 
frame_queue = Q(args.frame_queue_size) 

tube_extractor = TubeExtractor(args, tube_queue, frame_queue, running)
action_detector = ActionDetector(args, tube_queue, action_queue, running)
visualizer = Visualizer(args, action_queue, frame_queue, running)

if not (tube_extractor.valid and action_detector.valid):
	print('MAIN: error init!')
	sys.exit(0)

print('MAIN: init finished..')

threads = []
tube_thread = threading.Thread( target=tube_extractor.run )
threads.append(tube_thread)

action_thread = threading.Thread( target=action_detector.run )
threads.append(action_thread)

visualizer_thread = threading.Thread( target=visualizer.run )
threads.append(visualizer_thread)

for i in range(len(threads)):
	threads[i].setDaemon(True)
	threads[i].start()
print('MAIN: start running')

while True:
	try:
		sleep(1)
	except KeyboardInterrupt:
		running[0] = False
		print('MAIN: stopping...')
		sleep(0.5)
		break

print('done')