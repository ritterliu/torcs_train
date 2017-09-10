import os
import numpy as np
import base64
import scipy.misc
from StringIO import StringIO
from PIL import Image
import json
import glob
import math
import random
import wget
import math

PREFIX = 'https://cdn.cnbj1.fds.api.mi-img.com/'

class data_manager():
    def __init__(self, batch_size=100, label_size=2, log_set='', clip=0.5, fetch=False, prefix='', recovery_distance=4.0):
        self.batch_size = batch_size
        self.label_size = label_size
        self.clip = clip
        self.img_ori_height = 240
        self.img_width = 64
        self.img_height = 64
        self.fetch = fetch
        self.prefix = prefix
        self.recovery_distance = recovery_distance
        self.log_set = [ self.prefix + log for log in log_set.split()]
        self.local_log_set = []
        self.preprocess()

    def load_image(self, img_base64):
        img_str = base64.b64decode(img_base64)
        img_obj = StringIO(img_str)
        img = scipy.misc.imread(img_obj, mode='RGB')
        if img.shape != (240, 320, 3):
            print img.shape 
        if self.clip:
            clip_line = int(self.img_ori_height*self.clip)
            clip_line = self.img_ori_height - clip_line
            try:
                img = img[clip_line:, :, :]
            except IndexError:
                print 'img error', img.shape, clip_line
        img = Image.fromarray(img)
        img = img.resize((self.img_height, self.img_width))
        img = np.asarray(img, dtype='float32')
        img /= 255.0
        return img

    def fetch_from_fds(self):
        """ need to download from FDS using http URL """
        for log in self.log_set:
            if log.startswith('fds://'):
                print 'downloading', PREFIX + log[6:]
                filename = wget.download(PREFIX + log[6:])
                print 'downloaded to', filename
                self.local_log_set.append(filename)

    def preprocess(self):
        label_list = []
        # get all files within that folder
        for log in self.log_set:
            print 'read log', log
            if log.endswith('/'):
                self.log_set.remove(log)
                print 'after removed:', self.log_set
                self.log_set += glob.glob(log+'*')
                print 'after reappended:', self.log_set
        # if not using fuse, download FDS files to local
        if not self.fetch:
            self.local_log_set = self.log_set
        else:
            print 'downloading from FDS'
            self.fetch_from_fds()

        for logfile in self.local_log_set:
            print 'processing', logfile
            with open(logfile, 'r') as f:
                for line in f.readlines():
                    try:
                        frame_data = json.loads(line)
                    except ValueError:
                        print 'log file corrupted'
                        break
                    labels = [v for v in frame_data['control'].values()]
                    label_list.append(labels)

        labels = np.array(label_list)
        self.num_sample = labels.shape[0]
        print 'processing done, got raw samples:', self.num_sample
        self.steps_per_epoch = int(math.ceil(self.num_sample * 2.0 / self.batch_size)) # image duplicated

    def generator(self):
        train_features = np.zeros((self.batch_size, self.img_width, self.img_height, 3))
        train_labels = np.zeros((self.batch_size, self.label_size))
        cnt_train = 0
        while True:
            for logfile in self.local_log_set:
                with open(logfile, 'r') as f:
                    # 4: center, 5: left, 6: right
                    cam_type = int(logfile.split('__')[-1])
                    for line in f.readlines():
                        frame_data = json.loads(line)
                        img = self.load_image(frame_data['state']['screen'])
                        label_values = [v for v in frame_data['control'].values()]
                        brake = label_values[1]
                        accel = label_values[2]
                        steer = label_values[3]
                 
                        if brake > 0:
                            accel = -brake
                        # adjust steer by cam type
                        if cam_type == 5:  # left_cam
                            steer = steer + 1.0 / self.recovery_distance
                        elif cam_type == 6:
                            steer = steer - 1.0 / self.recovery_distance
                        # use steer and accel
                        label = [accel, steer]

                        train_features[cnt_train] = img
                        train_labels[cnt_train] = label
                        cnt_train += 1

                        if cnt_train < self.batch_size:
                            # duplicate
                            train_features[cnt_train] = np.fliplr(img)
                            train_labels[cnt_train] = [label[0], -label[1]]
                            cnt_train += 1

                        if cnt_train == self.batch_size:
                            idx = np.arange(self.batch_size)
                            np.random.shuffle(idx)
                            cnt_train = 0
                            yield (train_features[idx], train_labels[idx])
                        