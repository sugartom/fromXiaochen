# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""

import tensorflow as tf
slim = tf.contrib.slim


# The UCF-101 dataset has 101 classes
# NUM_CLASSES = 101

NUM_CLASSES = 40

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

WEIGHTS_PATH = './models/weights/'
import numpy as np


"-----------------------------------------------------------------------------------------------------------------------"

def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

# def inference_c3d(_X, _dropout, batch_size, _weights, _biases):
def inference_c3d(_X, _dropout, _weights, _biases):

  # Convolution Layer
  conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
  conv1 = tf.nn.relu(conv1, 'relu1')
  pool1 = max_pool('pool1', conv1, k=1)

  # Convolution Layer
  conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
  conv2 = tf.nn.relu(conv2, 'relu2')
  pool2 = max_pool('pool2', conv2, k=2)

  # Convolution Layer
  conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
  conv3 = tf.nn.relu(conv3, 'relu3a')
  conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
  conv3 = tf.nn.relu(conv3, 'relu3b')
  pool3 = max_pool('pool3', conv3, k=2)

  # Convolution Layer
  conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
  conv4 = tf.nn.relu(conv4, 'relu4a')
  conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
  conv4 = tf.nn.relu(conv4, 'relu4b')
  pool4 = max_pool('pool4', conv4, k=2)

  # Convolution Layer
  conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
  conv5 = tf.nn.relu(conv5, 'relu5a')
  conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
  conv5 = tf.nn.relu(conv5, 'relu5b')
  pool5 = max_pool('pool5', conv5, k=2)

  # Fully connected layer
  # pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
  # dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
  # import pdb;pdb.set_trace()
  dense1 = slim.flatten(pool5)
  dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

  dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)

  dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
  dense2 = tf.nn.dropout(dense2, _dropout)

  # Output: class prediction
  out = tf.matmul(dense2, _weights['out']) + _biases['out']

  # Bypass FCs
  # feat_shape = pool5.shape.as_list()
  # avg_feats = tf.nn.avg_pool3d(pool5, [1,1,feat_shape[2], feat_shape[3], 1], [1,1,feat_shape[2], feat_shape[3], 1], 'VALID')
  # avg_feats = slim.flatten(avg_feats)

  # drop1 = tf.nn.dropout(avg_feats, _dropout)
  # out = tf.matmul(drop1, _weights['out']) + _biases['out']

  # import pdb;pdb.set_trace()

  return out

def _variable_on_cpu(name, shape, initializer):
  cpu_id = 0
  with tf.device('/cpu:%d' % cpu_id):
  #with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(input_imgs, is_training):
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, NUM_CLASSES], 0.04, 0.005)
            # bypass fc
            # 'out': _variable_with_weight_decay('wout', [8192, NUM_CLASSES], 0.04, 0.005)
            # 'out': _variable_with_weight_decay('wout', [512, NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
            }

  dropout_keep = tf.cond(is_training, lambda: 0.2, lambda: 1.0)

  processed = preprocess(input_imgs)
  # processed = input_imgs
  
  logits = inference_c3d(processed, dropout_keep, weights, biases)

  # logits = tf.Print(logits,[dropout_keep], message='Dropout Value:')

  return logits


def initialize_weights(sess):
  var_to_restore = [var for var in tf.trainable_variables() if 'var_name' in var.name and 'out' not in var.name]
  # import pdb;pdb.set_trace()

  saver = tf.train.Saver(var_to_restore)
  saver.restore(sess, WEIGHTS_PATH + 'conv3d_deepnetA_sport1m_iter_1900000_TF.model')

def preprocess(input_seq):
  crop_mean = np.load('./action_detector/models/weights/c3d_crop_mean.npy')
  output_seq = input_seq - crop_mean

  # scale
  # output_seq = output_seq / 255.0
  return output_seq


if __name__ == '__main__':
    import numpy as np
    import pdb
    
    input_shape = [10, 16, 112, 112, 3]
    input_seq = tf.placeholder(tf.float32, input_shape, name='InputSequence')

    is_training = tf.placeholder(tf.bool, [], name='TrainFlag')

    logits = inference(input_seq, is_training)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)#,log_device_placement=True)
    sess = tf.Session(config=config)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    saver_vars = tf.global_variables()
    saver_vars = [var for var in saver_vars if 'out' not in var.name]
    saver = tf.train.Saver(saver_vars)

    saver.restore(sess, '/home/oytun/Dropbox/Python/Actions_AVA/models/weights/conv3d_deepnetA_sport1m_iter_1900000_TF.model')

    dummy_input = np.random.rand(*input_shape)
    pdb.set_trace()

    while True:
      logits_np = sess.run(logits, feed_dict={input_seq: dummy_input, is_training:False})

    # pdb.set_trace()

