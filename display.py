import cv2 
from time import time, sleep
import json 

NA = -1 

class Visualizer:
	def __init__(self, args, action_queue, frame_queue, running):
		self.stat = args.stat
		self.fps = args.video_fps
		self.show = args.visualize
		self.save_vid = args.save_vid 
		self.dst_vid = 'res.avi'

		self.colors = [ (127,255,127), (255,127,127), (127,127,255), 
						(64,127,255), (150,255,200), (200,150,255), 
						(255,255,0), (0,255,255), (255,0,255)]
		self.font = cv2.FONT_HERSHEY_SIMPLEX

		self.action_queue = action_queue
		self.frame_queue = frame_queue
		self.running = running
		self.frame_id = NA
		self.frame = None 
		self.frame_ts = None 

		with open('label_conversion.json') as fp:
			self.class_strs = json.load(fp)

		# fourcc = cv2.VideoWriter_fourcc(*'XVID')
		fourcc = cv2.cv.CV_FOURCC(*'XVID')
		self.vid_out = None 

		print('VIS: ready')
		self.valid = True 


	def read_frame(self):
		d = self.frame_queue.read()
		self.frame_id = d['frame_id']
		self.frame = d['frame']
		self.frame_ts = d['time']


	def run(self):
		fout = None 
		if self.stat:
			fout = open('stat.txt', 'w')

		while self.running[0]:
			acts = self.action_queue.read()
			fids = acts.keys()
			fids.sort()		# {fid: {id: [pos, acts]}}
			start_tid = fids[0]
			end_tid = fids[-1]

			if self.frame_id > end_tid:
				continue 

			for fid in fids:
				if self.frame_id < fid:
					self.read_frame()
				if self.frame_id > fid:
					continue

				self.draw_frame(acts[fid])
			
				if self.stat:
					fout.write('%d %f\n' % (self.frame_id, time() - self.frame_ts))

				if self.save_vid:
					if self.vid_out == None:
						h, w, _ = self.frame.shape
						# fourcc = cv2.VideoWriter_fourcc(*'XVID')
						fourcc = cv2.cv.CV_FOURCC(*'XVID')
						self.vid_out = cv2.VideoWriter(self.dst_vid, fourcc, self.fps, (w, h))
					self.vid_out.write(self.frame)

				if self.show:
					cv2.imshow('res', self.frame)

				if self.running[0] == False or cv2.waitKey(1) & 0xFF == ord('q'):
					self.running[0] = False 
					break

				sleep(1. / self.fps)

		print('VIS: ending...')

		if self.stat:
			fout.close()
		if self.save_vid:
			self.vid_out.release()
		print('VIS: ended')


	def action_name(self, pos):
		return self.class_strs['training2real'][str(pos)][1]


	def draw_frame(self, act):
		for i in act:
			pos, acts = act[i]

			left = pos[0]
			top = pos[1]
			right = pos[2]
			bottom = pos[3]

			cv2.rectangle(self.frame, (left, top), (right, bottom),
							self.colors[i % len(self.colors)], thickness=2)

			txt = `i`
			best_action = ''
			best_prob = -1.
			for a in range(len(acts)):
				if acts[a] > best_prob:
					best_prob = acts[a]
					best_action = self.action_name(a)

			if best_prob > 0:
				txt += ': %s(%0.2f)' % (best_action, best_prob)

			cv2.putText(self.frame, txt, (left, top - 12), self.font, fontScale=0.7,
							color=self.colors[i % len(self.colors)], thickness=2)

			if self.stat:
				cv2.putText(self.frame, 
							'Frame: %d - Latency: %ds' % (self.frame_id, 
														time() - self.frame_ts), 
							(20, 30), self.font, fontScale=1,
							color=(255,255,0), thickness=2)
