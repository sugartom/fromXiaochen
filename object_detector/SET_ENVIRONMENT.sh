source /home/oytun/python/envs/latestTF/bin/activate
TF_MODELS_PATH="/home/oytun/Dropbox/Python/AVA/object_detection/models"
export PYTHONPATH=$PYTHONPATH:$TF_MODELS_PATH:$TF_MODELS_PATH/slim:$TF_MODELS_PATH/object_detection
echo $PYTHONPATH
