https://blog.csdn.net/qq_21368481/article/details/89448226

sudo apt-get install python3.6-dev

https://github.com/mcfletch/pyopengl/issues/11


'->exc_' replace '->curexc_'

pip install --upgrade setuptools


virtualenv --python=/usr/bin/python3.6 python3.6


<!-- python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/DOTA/'  -->

python convert_data_to_tfrecord.py --VOC_dir='/fdisk/data/dataset/DOTA/DOTA1.0/trainval/' --xml_dir='labeltxt' --image_dir='images' --save_name='train'  --img_format='.png'  --dataset='DOTA'

python multi_gpu_train_r3det.py

python test_dota_r3det.py --test_dir='/fdisk/DOTA/val/images/' --gpus=0

python demo_dota_r3det.py --test_dir='/fdisk/DOTA/val/images/' --gpus=0

python test_dota_r3det.py --test_dir='/fdisk/DOTA/smalltest/' --gpus=0

python demo_dota_r3det.py --test_dir='/fdisk/DOTA/smalltest/' --gpus=0


export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:"$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"


cfg.py
CLASS_NUM = 16
label_dict.py
cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15,
        'container-crane':16
    }