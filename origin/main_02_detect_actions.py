import tensorflow as tf
import numpy as np
import json
import os
import cv2
import argparse
from time import time 
import action_detector.models.c3d_model as c3d_model

# CKPT_FILE = './action_detector/models/weights/model_ckpt-19'
CKPT_FILE = './action_detector/models/weights/model_ckpt-3'
INPUT_SHAPE = [16,112,112,3]

BATCH_SIZE = 6

with open('label_conversion.json') as fp:
    LABEL_CONV = json.load(fp)
    # "training2real": {"0": [15, "answer_phone"], ...
    # "real2training": {"15": [0, "answer_phone"], ...


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=str, required=True)
    parser.add_argument('-v', '--visualize_video', type=int, required=False, default=1)
    

    args = parser.parse_args()

    gpu = args.gpu
    input_folder = args.input_folder
    visualize_flag = bool(args.visualize_video)

    detect_actions(gpu, input_folder, visualize_flag)


def detect_actions(gpu, input_folder, visualize_flag):
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-i', '--input_folder', type=str, required=True)
    # parser.add_argument('-g', '--gpu', type=str, required=True)
    # parser.add_argument('-v', '--visualize_video', type=int, required=False, default=1)
    

    # args = parser.parse_args()

    print('\nRunning action detection!\n')

    # gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # input_folder = args.input_folder
    input_folder = input_folder[:-1] if input_folder[-1]=='/' else input_folder
    video_name = input_folder.split('/')[-1]

    # visualize_flag = bool(args.visualize_video)
    print('Visualization: %s' % ('yes' if visualize_flag else 'no'))


    # graph def
    is_training = tf.constant(False)
    input_seq = tf.placeholder(tf.float32, [None] + INPUT_SHAPE, name='InputSequence')


    model = c3d_model
    logits = model.inference(input_seq, is_training)

    pred_probs = tf.nn.sigmoid(logits)
    pred_probs = tf.clip_by_value(pred_probs, 1e-5, 1 - 1e-5)

    init_op = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # load checkpoints
    model_saver.restore(sess, CKPT_FILE)
    print('Loading model checkpoint from: ' + CKPT_FILE)

    gen = data_input(input_folder)
    
    prev_tube = ''
    
    results_dict = {}
    total_time = time()
    for sample, tube in gen:
        sample_exp = np.expand_dims(sample, axis=0)

        class_probs = sess.run(pred_probs, feed_dict={input_seq:sample_exp})
        class_probs = class_probs[0]
        class_probs = [get_3_decimal_float(prob) for prob in class_probs]

        tube_name = tube.split('.')[0]
        if tube_name in results_dict.keys():
            results_dict[tube_name].append(class_probs)
        else:
            results_dict[tube_name] = [class_probs]

        if tube != prev_tube:
            print('Avg action detection: %0.2f' % ((time() - total_time)))
            total_time = time()
            print('Working on %s' %tube)
            prev_tube = tube

        if visualize_flag: 
            visualize_sample_and_result(sample, class_probs)


    actions_json_path = './jsons/%s_actions.json' % video_name

    with open(actions_json_path, 'w') as fp:
        json.dump(results_dict, fp)
    
    print('Saved actions to %s!' % actions_json_path)

    sess.close()
    # import pdb;pdb.set_trace()

        


def get_next(vcap):
    ret1, frame = vcap.read()
    ret2, frame = vcap.read()
    if not(ret1 and ret2):
        return None, False
    else:
        reshaped = cv2.resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[2]))
        return reshaped, True

def data_input(tubes_folder):
    tubes_list = os.listdir(tubes_folder)
    tubes_list.sort()
    
    [timesteps, H, W, C] = INPUT_SHAPE
    # slope = (no_images-1) / float(timesteps - 1)
    # indices = (slope * np.arange(timesteps)).astype(np.int64)

    for tube in tubes_list:
        video_path = os.path.join(tubes_folder, tube)

        vcap = cv2.VideoCapture(video_path)
        batch_np = np.zeros([timesteps, H, W, C], np.uint8)
        overlap = timesteps // 2
        tube_done = False
        while not tube_done:
            batch_np = np.concatenate([batch_np[overlap:, :,:,:], 
                                       np.zeros([timesteps-overlap, H, W, C], np.uint8)], 
                                       axis=0)
            # import pdb; pdb.set_trace()
            for tt in range(overlap, timesteps):
                frame, ret = get_next(vcap)
                if not ret:
                    tube_done = True
                    break
                batch_np[tt,:,:,:] = frame

            yield batch_np, tube
            
def get_3_decimal_float(infloat):
    return float('%.3f' % infloat)

def visualize_sample_and_result(sample, class_probs=None):
    T, H, W, C = sample.shape
    img_to_show = np.zeros([H,T*W,C], np.uint8)

    for t in range(T):
        start_idx = t * W
        end_idx = (t+1) * W
        img_to_show[:,start_idx:end_idx,:] = sample[t, :, :, :]

    if class_probs:
        class_probs = np.array(class_probs)
        class_order = np.argsort(class_probs) [::-1]
        probs_order = class_probs[class_order]
        class_strs = [LABEL_CONV['training2real'][str(class_no)][1] for class_no in class_order]

        printable = [ '%s : %.3f' % (class_str, prob) for class_str, prob in zip(class_strs,probs_order)]
        print(printable)

        no_print = 5
        pixel_distance = 25
        black_bar = np.zeros([(no_print+2) * pixel_distance, T * W,C], np.uint8)

        for ii in range(no_print):
            cur_printable = printable[ii]
            cv2.putText(black_bar, cur_printable, (T * W // 2, (ii+2) * pixel_distance), 0, 1, (255,255,255), 1)

        img_to_show = np.concatenate([img_to_show, black_bar], axis = 0)


    cv2.imshow('results', img_to_show)
    k = cv2.waitKey(0)
    if k == ord('q'):
        os._exit(0)



if __name__ == '__main__': 
    main()
# if __name__ == '__main__': 
#     gen = data_input('./tubes/virat_1')
#     for batch, tube in gen:
#         # import pdb;pdb.set_trace()
#         visualize_sample_and_result(batch)
#         # print(tube)
#         pass



