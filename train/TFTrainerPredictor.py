# -*- coding: utf-8 -*-
from __future__ import division
import glob
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import json
from Config import *
from DataProcessor import DataProcessor
import copy

class nvidia_net:
    def __init__(self):
        ops.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.dp = DataProcessor(DataConfig())
        self.config = self.dp.config
        self.row = self.config.img_height
        self.col = self.config.img_width
        self.ch = self.config.num_channels
        self.batch_count = 0
        self.create_nvidia_model()
        self.create_train_method()
        self.epoch_count = 0
        self.step_count = 0
        self.loss_val = 1
        self.saver = tf.train.Saver()
        if self.config.model_continue:
            self.restore_sess()
        else:
            self.sess.run(tf.global_variables_initializer())

    def data_generator_one_by_one(self):
        single_log_paths = sorted(glob.glob("{}/*".format(self.config.train_data_path)))
        index = 0
        while True:
            self.batch_count = 0
            for single_log_path in single_log_paths:
                print 'one by one, single_log_path:', single_log_path
                batch_features = np.zeros((self.config.batch_size, int(self.row * self.config.clip), self.col, self.ch),
                                          dtype=np.float32)
                batch_labels = np.zeros((self.config.batch_size, self.config.label_size), dtype=np.float32)
                for line in open(single_log_path):
                    img = self.dp.get_X(json.loads(line))
                    batch_features[index, :, :, :] = img
                    batch_labels[index] = self.dp.get_real_label(json.loads(line))
                    if index == self.config.batch_size - 1:
                        self.batch_count += 1
                        yield (batch_features, batch_labels)
                    index = (index + 1) % self.config.batch_size
                print 'one by one, save_sess()'
            self.save_sess()
            self.epoch_count += 1

    def data_generator_one_by_one2(self):
        print 'one by one...'
#        total_num_rows, self.num_dic = self.dp.count_data(self.config.train_data_path, )
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
            #print 'one by one index_list:', index_list
            for index in index_list:
                single_log_path = single_log_paths[index]
                #print 'one by one index:', index, ',single_log_path:', single_log_path
                line = f_dic[single_log_path].readline()
                num_dic[single_log_path] -= 1
                if num_dic[single_log_path] == 0:
                    num_dic[single_log_path] = self.num_dic[single_log_path]
                    f_dic[single_log_path] = open(single_log_path)
                img = self.dp.get_X(json.loads(line))
                batch_features[batch_index, :, :, :] = img
                batch_labels[batch_index] = self.dp.get_real_label(json.loads(line))
                batch_index = (batch_index + 1) % self.config.batch_size
            assert batch_index == 0
            self.step_count += 1
            #print 'one by one2...before yield step:', self.step_count, '/', self.steps_per_epoch
            yield (batch_features, batch_labels)

    def print_loss(self, total_num_rows, batch):
        batches_per_seg = 1
        seg = int(total_num_rows/self.config.batch_size/batches_per_seg)
        if self.batch_count % batches_per_seg != 0:
            return
        percent = self.batch_count * self.config.batch_size / total_num_rows
        #print self.batch_count * self.config.batch_size, '/', total_num_rows, ' , percent:', percent
        self.loss_val = self.loss.eval(feed_dict={
            self.img_input: batch[0],
            self.label_input: batch[1]
        })
        print 'print_loss : step/steps_per_epoch:', self.step_count,'/', self.steps_per_epoch, ', loss:', self.loss_val

    def train_network(self):
        #total_num_rows, _ = self.dp.count_data(self.config.train_data_path)
        total_num_rows, self.num_dic = self.dp.count_data(self.config.train_data_path, )
        self.steps_per_epoch=int(total_num_rows / self.config.batch_size)
        print 'train_network self.steps_per_epoch:' , self.steps_per_epoch
        generator = self.data_generator_one_by_one2()
        last_loss = 1
        while True:
            batch = generator.next()
            self.optimizer.run(feed_dict={
                self.img_input: batch[0],
                self.label_input: batch[1]
            })
            self.print_loss(total_num_rows, batch)

            if self.step_count >= self.steps_per_epoch:
                self.step_count = 0
                if self.loss_val < last_loss:
                    print 'train_network self.loss<last_loss:', self.loss_val, '<', last_loss, ' save modle'
                    last_loss = self.loss_val
                    self.save_sess()
                    self.epoch_count += 1
                else:
                    print 'train_network self.loss_val >= last_loss:', self.loss_val, ' >= ', last_loss, ' do not save model'

                if self.epoch_count >= self.config.epoch:
                    print 'train_network self.epoch_count >= self.config.epoch return'
                    return
                else:
                    print 'train_network next...self.epoch_count:', self.epoch_count


    def create_train_method(self):
        self.label_input = tf.placeholder('float', [None, self.config.label_size])
        self.loss = tf.reduce_mean(tf.squared_difference(self.nvidia_output, self.label_input))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def create_nvidia_model(self):
        def build_layers(input_imgs, w_i, b_i):
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv2d(input_imgs, filters=24, kernel_size=[5, 5], strides=[2, 2],
                                         padding='SAME', activation=tf.nn.relu,
                                         kernel_initializer=w_i, bias_initializer=b_i)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv2d(conv1, filters=36, kernel_size=[5, 5], strides=[2, 2],
                                         padding='SAME', activation=tf.nn.relu,
                                         kernel_initializer=w_i, bias_initializer=b_i)
            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv2d(conv2, filters=48, kernel_size=[3, 3], strides=[2, 2],
                                         padding='SAME', activation=tf.nn.relu,
                                         kernel_initializer=w_i, bias_initializer=b_i)
            with tf.variable_scope('conv4'):
                conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                         padding='SAME', activation=tf.nn.relu,
                                         kernel_initializer=w_i, bias_initializer=b_i)
            with tf.variable_scope('conv5'):
                conv5 = tf.layers.conv2d(conv4, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                         padding='SAME', activation=tf.nn.relu,
                                         kernel_initializer=w_i, bias_initializer=b_i)
            with tf.variable_scope('dense1'):
                dense1 = tf.layers.dense(tf.contrib.layers.flatten(conv5), activation=tf.nn.relu, units=50)
            with tf.variable_scope('dense2'):
                dense2 = tf.layers.dense(dense1, activation=tf.nn.relu, units=10)
            with tf.variable_scope('dense3'):
                dense3 = tf.layers.dense(dense2, units=self.config.label_size)
            return dense3

        self.img_input = tf.placeholder("float", [None, int(self.row * self.config.clip), self.col, self.ch])
        with tf.variable_scope('nvidia_net'):
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            self.nvidia_output = build_layers(self.img_input, w_i, b_i)

    def save_sess(self):
        self.saver.save(self.sess, self.config.tensorflow_model_path)
        print 'Sess saved in {} !!!'.format(self.config.tensorflow_model_path)

    def restore_sess(self):
        self.saver.restore(self.sess, self.config.tensorflow_model_path)
        print 'Sess restored from {} !!!'.format(self.config.tensorflow_model_path)

    def predict(self, img_json):
        img = np.zeros((1, int(self.row * self.config.clip), self.col, self.ch), dtype=np.float32)
        img[0] = self.dp.get_X(img_json)
        return self.nvidia_output.eval(feed_dict={self.img_input: img})[0]

    def control(self):
        log_jsons = sorted(glob.glob("{}/*".format(self.config.predict_data_path)))
        for log_jsons_path in log_jsons:
            with open(log_jsons_path) as f:
                for line in f.readlines():
                    print(self.predict(json.loads(line)))
                    print 'real:', self.dp.get_real_label(json.loads(line)), '\n'


if __name__ == "__main__":
    trainer = nvidia_net()
    # trainer.control()
    trainer.train_network()







