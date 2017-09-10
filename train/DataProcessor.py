# -*- coding: utf-8 -*-
import glob
import base64
import scipy.misc
from StringIO import StringIO
import os
import numpy as np
import json
import copy

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.num_channels = config.num_channels
        self.row = config.img_height
        self.col = config.img_width
        self.ch = config.num_channels

    def load_model(self):
        model_path = self.config.keras_model_path
        print os.path.abspath(os.curdir)
        print os.path.split(os.path.realpath(__file__))[0]
        from keras.models import load_model
        model = load_model(model_path)
        return model

    def get_X(self, img_json, mode='RGB'):
        assert mode == 'RGB' or mode == 'GRAY'
        img_str = base64.b64decode(img_json['state']['screen'])
        img_buff = StringIO(img_str)
        img = scipy.misc.imread(img_buff, mode='RGB')
        img = img.astype('float') / 255.0
        return img[self.row - int(self.row * self.config.clip):, :, :]

    def get_real_label(self, fds):
        infos = [(k, v) for k, v in fds['state'].iteritems() if k != 'screen']
        infos += [(k, v) for k, v in fds['groundtruth'].iteritems()]
        infos += [(k, v) for k, v in fds['control'].iteritems()]
        infos = {k: v for (k, v) in infos}

        brake_accel = -1.0 * infos['brake'] + 1.0 * infos['accel']
        if self.config.label_mode == "3-accel-brake_steer_offset":
            return [brake_accel, infos['steer'], infos['offset']]
        if self.config.label_mode == "1-steer":
            return [infos['steer']]
        assert False

    def count_data(self, logs_path):
        single_log_paths = sorted(glob.glob("{}/*".format(logs_path)))
        total_num_rows = 0
        num_dic = {}
        for single_log_path in single_log_paths:
            print 'single_log_path:', single_log_path
            num_rows = 0
            for _ in open(single_log_path):
                num_rows += 1
            print 'num_rows:', num_rows
            total_num_rows += num_rows
            num_dic[single_log_path] = num_rows
        print 'total_num_rows  ', total_num_rows
        return total_num_rows, num_dic

    def data_generator(self):
        single_log_paths = sorted(glob.glob("{}/*".format(self.config.train_data_path)))
        batch_features = np.zeros((self.config.batch_size, int(self.row * self.config.clip), self.col, self.ch),
                                  dtype=np.float32)
        batch_labels = np.zeros((self.config.batch_size, self.config.label_size), dtype=np.float32)
        batch_index = 0
        while True:
            for single_log_path in single_log_paths:
                for line in open(single_log_path, 'rb'):
                    img = self.get_X(json.loads(line))
                    batch_features[batch_index, :, :, :] = img
                    batch_labels[batch_index] = self.get_real_label(json.loads(line))
                    batch_index = (batch_index + 1) % self.config.batch_size
                    if batch_index == 0:
                        yield (batch_features, batch_labels)

    def data_generator_one_by_one(self):
        print 'one by one...'
        total_num_rows, self.num_dic = self.count_data(self.config.train_data_path, )
        num_dic = copy.deepcopy(self.num_dic)
        single_log_paths = sorted(glob.glob("{}/*".format(self.config.train_data_path)))
        f_dic = {}
        for single_log_path in single_log_paths:
            f_dic[single_log_path] = open(single_log_path)
        batch_index = 0
        batch_features = np.zeros((self.config.batch_size, int(self.row * self.config.clip), self.col, self.ch),
                                  dtype=np.float32)  # 裁剪，只取下半部分
        batch_labels = np.zeros((self.config.batch_size, self.config.label_size), dtype=np.float32)
        while True:
            index_list = np.random.randint(0, len(single_log_paths), size=self.config.batch_size)
            print 'one by one index_list:', index_list
            for index in index_list:
                single_log_path = single_log_paths[index]
                #print 'index:', index, ', single_log_path:', single_log_path
                line = f_dic[single_log_path].readline()
                num_dic[single_log_path] -= 1
                if num_dic[single_log_path] == 0:
                    num_dic[single_log_path] = self.num_dic[single_log_path]
                    f_dic[single_log_path] = open(single_log_path)
                img = self.get_X(json.loads(line))
                batch_features[batch_index, :, :, :] = img
                batch_labels[batch_index] = self.get_real_label(json.loads(line))
                batch_index = (batch_index + 1) % self.config.batch_size
            assert batch_index == 0
            print 'going to yield....'
            yield (batch_features, batch_labels)
