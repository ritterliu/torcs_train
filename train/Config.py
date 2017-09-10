# -*- coding: utf-8 -*-
import os


class DataConfig(object):
    use_fds = True #True
    use_torcs_bucket = False #Tru
    bucket = os.environ.get('XIAOMI_FDS_DEFAULT_BUCKET') or 'johndoe'
   
    # you can upload your own data to FDS for training
    if use_fds:
        prefix = "/fds"
    # local train
    else:
        prefix = os.path.split(os.path.realpath(__file__))[0]

    # by default, the official 14G data is mounted to /torcs in the torcs-keras docker
    if use_torcs_bucket:
        train_data_path = "/torcs/camera_data"
        # using custom validation set if using camera_data
        val_data_path = '/fds/camera_data/raw_val'
        # every one has right to read /torcs but not write, so you have to write in your own fds bucket
        model_path = "/fds"
    else:
        data_path = prefix + "/camera_data/"
        train_data_path = data_path + "raw_train"
        val_data_path = data_path + "raw_val"
        predict_data_path = data_path + "raw_predict"
        model_path = prefix

    train_data_name = "raw_train"
    val_data_name = "raw_val"
    predict_data_name = "raw_predict"

    img_height = 240
    img_width = 320
    num_channels = 3
    epoch = 1000

    model_continue = False

    label_mode = "3-accel-brake_steer_offset"
    # label_mode = :"1-steer"
    label_size = int(label_mode.split('-')[0])

    clip = 0.6  # 图片的下百分之75部分用来训练

    batch_size = 64

    model_type = 'tensorflow'
    keras_model_path = model_path + "/keras_model/example_model_for_wheel-2"
    tensorflow_model_path = model_path + '/tensorflow_model/model.ckpt'
