#use it with source SET_ENVIRONMENT.sh
CUR_PWD=$PWD
OBJ_PATH=$CUR_PWD/object_detector
ACT_PATH=$CUR_PWD/action_detector
TF_MODELS_PATH=$OBJ_PATH/models
export PYTHONPATH=$PYTHONPATH:$OBJ_PATH:$ACT_PATH:$TF_MODELS_PATH:$TF_MODELS_PATH/slim:$TF_MODELS_PATH/object_detection
echo $PYTHONPATH

#source /home/oytun/python/envs/latestTF/bin/activate

