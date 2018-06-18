import numpy as np 
from time import time 
import action_detector.models.c3d_model as c3d_model
import os 
import tensorflow as tf
import cv2 

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
# Yitao-TLS-End

class ActionDetector:

	def __init__(self, args, tube_queue, action_queue, running):

		self.batch = args.batch 
		self.video_fps = args.video_fps
		self.ckpt_file = args.action_ckpt
		self.min_tube_len = args.min_tube_len

		self.input_shape = [16,112,112,3]
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

		self.tube_queue = tube_queue
		self.action_queue = action_queue
		self.running = running

		# init TF
		is_training = tf.constant(False)
		self.input_seq = tf.placeholder(tf.float32, [None] + self.input_shape, name='InputSequence')

		self.model = c3d_model
		logits = self.model.inference(self.input_seq, is_training)

		self.pred_probs = tf.nn.sigmoid(logits)
		self.pred_probs = tf.clip_by_value(self.pred_probs, 1e-5, 1 - 1e-5)

		init_op = tf.global_variables_initializer()
		model_saver = tf.train.Saver()

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.sess = tf.Session(config=config)

		# load checkpoints
		model_saver.restore(self.sess, self.ckpt_file)
		print('ACTION: Loading model checkpoint from: ' + self.ckpt_file)

		self.valid = True 
		print('ACTION: init done')

		# # Yitao-TLS-Begin
		# export_path_base = "c3d_tensorflow"
		# export_path = os.path.join(
		# 	compat.as_bytes(export_path_base),
		# 	compat.as_bytes(str(1)))
		# print 'Exporting trained model to', export_path
		# builder = saved_model_builder.SavedModelBuilder(export_path)

		# tensor_info_x = tf.saved_model.utils.build_tensor_info(self.input_seq)
		# tensor_info_y = tf.saved_model.utils.build_tensor_info(self.pred_probs)

		# prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
		# 	inputs={'input': tensor_info_x},
		# 	outputs={'output': tensor_info_y},
		# 	method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

		# legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
		# builder.add_meta_graph_and_variables(
		# 	self.sess, [tf.saved_model.tag_constants.SERVING],
		# 	signature_def_map={
		# 		'predict_images':
		# 			prediction_signature,
		# 	},
		# 	legacy_init_op=legacy_init_op)

		# builder.save()

		# print('Done exporting!')
		# # Yitao-TLS-End


	def dummy_detect_action(self, tubes):
		acts = {}
		# print(tubes)
		for fid in tubes:
			acts[fid] = {}
			for people in tubes[fid]:
				pos = tubes[fid][people][0]
				acts[fid][people] = [pos, []]
		return acts


	def transform(self, tubes):
		res = {}
		for f in tubes:
			for i in tubes[f]:
				if not i in res:
					res[i] = []
				res[i].append(tubes[f][i])

		return res 


	def data_input(self, tubes_in):
		[timesteps, H, W, C] = self.input_shape

		# # print("[Yitao] self.input_shape = %s" % str(self.input_shape))
		# # print("[Yitao] tubes_in.keys() = %s" % str(tubes_in.keys()))
		# check_keys = [1, 2]
		# for check_key in check_keys:
		# 	if (check_key in tubes_in):
		# 		for i in range(1, 4):
		# 			print("[Yitao] tubes_in[%d][%d][0] = %s" % (check_key, i, str(tubes_in[check_key][i][0])))
		# 			print("[Yitao] tubes_in[%d][%d][1].shape = %s" % (check_key, i, str(tubes_in[check_key][i][1].shape)))
		# 		# print("[Yitao] tubes_in[1][1][0] = %s" % str(tubes_in[1][1][0]))
		# 		# print("[Yitao] tubes_in[1][1][1].shape = %s" % str(tubes_in[1][1][1].shape))
		# 		# print("[Yitao] tubes_in[1][2][0] = %s" % str(tubes_in[1][2][0]))
		# 		# print("[Yitao] tubes_in[1][2][1].shape = %s" % str(tubes_in[1][2][1].shape))
		# 		# print("[Yitao] tubes_in[1][3][0] = %s" % str(tubes_in[1][3][0]))
		# 		# print("[Yitao] tubes_in[1][3][1].shape = %s" % str(tubes_in[1][3][1].shape))
		# 	# print("[Yitao] tubes_in[1][2].len = %s" % str(len(tubes_in[1][2])))
		# 	# print("[Yitao] tubes_in[1][3].len = %s" % str(len(tubes_in[1][3])))

		tubes = self.transform(tubes_in)

		res = {}
		for i in tubes:
			tube = tubes[i]
			res[i] = []
			
			if len(tube) < self.min_tube_len:
				continue

			batch_np = np.zeros([timesteps, H, W, C], np.uint8)
			overlap = timesteps // 2  
			step =  timesteps - overlap
			for f in range(len(tube)):
				if f % step == 0 or f == len(tube) - 1:
					if f > 0: 
						res[i].append(batch_np.copy())
					batch_np = np.concatenate(	[batch_np[overlap:, :,:,:], 
												np.zeros([step, H, W, C], np.uint8)], 
												axis=0)

				pos, frame = tube[f]
				reshaped = cv2.resize(frame, (self.input_shape[1], self.input_shape[2]))
				batch_np[overlap + f % step,:,:,:] = reshaped

		# format: {id: [batch1, batch2, ...]}
		return res


	def data_output(self, acts, tubes):
		# acts: {id: [act_list1, act_list2, ...]}
		# tubes: {fid: {id: [pos, frame]}}
		# res: {fid: {id: [pos, action_list}}
		step = self.input_shape[0]/2
		id_cnt = {}

		res = {}
		for fid in tubes:
			res[fid] = {}
			for i in tubes[fid]:
				if not i in acts or len(acts[i]) == 0:
					continue

				if not i in id_cnt:
					id_cnt[i] = 0
				id_cnt[i] += 1

				pos, frame = tubes[fid][i]
				act_pos = min(id_cnt[i] % step, len(acts[i]) - 1)
				res[fid][i] = [pos, acts[i][act_pos]]

		return res 


	def get_3_decimal_float(self, infloat):
		return float('%.3f' % infloat)
	

	def detect_action(self, tubes):
		data = self.data_input(tubes)	
		
		res = {}
		for i in data:
			res[i] = []

			for sample in data[i]:
				sample_exp = np.expand_dims(sample, axis=0)

				class_probs = self.sess.run(self.pred_probs, feed_dict={self.input_seq:sample_exp})
				class_probs = class_probs[0]
				class_probs = [self.get_3_decimal_float(prob) for prob in class_probs]

				res[i].append(class_probs)

		return self.data_output(res, tubes)


	def run(self):
		while self.running[0]:	
			tubes = self.tube_queue.read()
			cur_time = time()	
			# acts = self.dummy_detect_action(tubes)
			acts = self.detect_action(tubes)

			# format of acts: {fid: {id: [pos, action_list}}
			self.action_queue.write(acts)
			print('ACTION: FPS: %d' % (self.batch * self.video_fps / (time() - cur_time)))

		print('ACTION: Ending...')		
		self.sess.close()
		print('ACTION: Ended')		

