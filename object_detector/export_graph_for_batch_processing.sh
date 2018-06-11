# make sure object_detection api is built using: protoc object_detection/protos/*.proto --python_out=.
# .config files are located in models/object_detection/samples/configs/
MODEL_PATH=/home/oytun/Dropbox/Python/AVA/object_detection/zoo/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017
python ./models/object_detection/export_inference_graph.py --input_type image_tensor \
							   --pipeline_config_path $MODEL_PATH/faster_rcnn_inception_resnet_v2_atrous_coco.config \
							   --trained_checkpoint_prefix $MODEL_PATH/model.ckpt \
							   --output_directory $MODEL_PATH/out_graph
