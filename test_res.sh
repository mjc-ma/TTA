export PYTHONPATH=$PYTHONPATH:$(pwd)


# Resnet50
# CUDA_VISIBLE_DEVICES=1 python model_adapt.py --data_dir /data/majc/imagenet-s
CUDA_VISIBLE_DEVICES=1 python model_adapt.py --data_dir /data/majc/imagenet-r


