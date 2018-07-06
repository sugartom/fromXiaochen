import cv2
from time import time
import os 
from object_detector.yolo_new import YOLO
from tracker.tracker import Tracker

NA = -1 

class TubeHolder:	
	def __init__(self):
		self.tubes = {}


	def add_tks(self, tks, img, fid):
		# format of tracks:
		# [
		#   [x0, y0, x1, y1, tk_id], ...
		# ]
		self.tubes[fid] = {}

		for tk in tks:
			tk = map(int, tk)
			people_id = tk[-1]
			pos = self.box_in_frame(tk, img)
			crop = self.crop_frame(pos, img)
			self.tubes[fid][people_id] = [pos, crop]


	def box_in_frame(self, tk, img):
		h, w, _ = img.shape
		left = max(0, tk[0])
		top = max(0, tk[1])
		right = min(w - 1, tk[2])
		bottom = min(h - 1, tk[3])
		return [left, top, right, bottom]


	def crop_frame(self, pos, img):
		return img[pos[1]:pos[3], pos[0]:pos[2]]


	def summary(self):		# show the summary of the batch {id: vid_len}
		s = {}
		for f in self.tubes:
			for i in self.tubes[f]:
				if not i in s:
					s[i] = 0
				s[i] += 1 
		return s 


	def clear(self):
		self.tubes = {}
		

class TubeExtractor:
	def __init__(self, args, tube_queue, frame_queue, running):
		
		self.yolo = YOLO(args)
		self.tracker = Tracker(args.max_age, args.min_hits)

		self.max_frame = args.max_frame
		self.video_fps = args.video_fps
		self.batch = args.batch

		self.tube_queue = tube_queue
		self.frame_queue = frame_queue
		self.running = running

		src = args.video_src
		if src.isdigit():
			src = int(src)
		elif not os.path.exists(src):
			print('TUBE: cannot read video!')
			return

		self.cap = cv2.VideoCapture(src)
		
		self.tube_holder = TubeHolder()

		self.valid = True


	def get_fps(self):
		pass 
		# return self.


	def dummy_tracker(self, dets):
		res = []
		cnt = 0
		for d in dets:
			res.append(d[:4] + [cnt])
			cnt += 1
		return res 


	def run(self):
		frame_id = 0
		
		total_time = 0
		while self.running[0]:
			ret, frame = self.cap.read()
			if not ret:
				print('TUBE: cannot read frame!')
				self.running[0] = False
				break

			frame_id += 1
			if frame_id > self.max_frame and self.max_frame > 0:
				print('TUBE: Exceeds max frame id %d' % self.max_frame)
				self.running[0] = False
				break

			cur_time = time()

			det = self.yolo.detect_frame(frame)
			det_people, det_obj = self.yolo.parse_det(det)
			
			# print(det_people)

			for i in range(len(det_people)):
				det_people[i] = det_people[i][:4]

			tks = self.tracker.update(det_people)			
			# tks = self.dummy_tracker(det_people)	# use det res as tracking res
			
			self.frame_queue.write({'frame_id': frame_id, 
									'frame': frame, 
									'time': cur_time})

			self.tube_holder.add_tks(tks, frame, frame_id)
			total_time += time() - cur_time

			if frame_id % (self.batch * self.video_fps) == 0:
				if frame_id == 0:
					continue 

				print('TUBE: FPS %d' % (self.batch * self.video_fps / total_time))
				total_time = 0
				# print('tubes: %s' % `self.tube_holder.summary()`)
				self.tube_queue.write(self.tube_holder.tubes.copy())
				self.tube_holder.clear()


		print('TUBE: stopping...')
		self.cap.release()
		print('TUBE: ended')