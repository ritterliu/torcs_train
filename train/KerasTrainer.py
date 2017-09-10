# -*- coding: utf-8 -*-
import time
import gc
import glob
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.callbacks import ModelCheckpoint
from Config import *
from DataProcessor import DataProcessor
import numpy as np
import json


class CNNTrainer:
    def __init__(self, DataProcessor):
        self.dp = DataProcessor
        self.config = self.dp.config
        self.row = self.config.img_height
        self.col = self.config.img_width
        self.ch = self.config.num_channels
        self.X_val, self.y_val = None, None
        self.get_val_data()
        self.callback_list = self.get_callback_list()
        self.model = self.dp.load_model() if self.config.model_continue else self.simple_model()

    def get_val_data(self):
        single_log_paths = sorted(glob.glob("{}/*".format(self.config.val_data_path)))
        print 'single_log_paths:', single_log_paths
        if len(single_log_paths) == 0:
            return
        val_count = reduce(lambda x, y: x+y, [len(open(p, 'rb').readlines()) for p in single_log_paths])
        self.X_val = np.zeros((val_count, int(self.row * self.config.clip), self.col, self.ch), dtype=np.float32)
        self.y_val = np.zeros((val_count, self.config.label_size), dtype=np.float32)
        index = 0
        for single_log_path in single_log_paths:
            for line in open(single_log_path, 'rb'):
                img = self.dp.get_X(json.loads(line))
                self.X_val[index, :, :, :] = img
                self.y_val[index] = self.dp.get_real_label(json.loads(line))
                index += 1

    def train(self):
        print 'train_path:', self.config.train_data_path
        total_num_rows, _ = self.dp.count_data(self.config.train_data_path, )
        self.model.fit_generator(self.dp.data_generator_one_by_one(),
                                 verbose=1,
                                 steps_per_epoch=int(total_num_rows / self.config.batch_size),
                                 validation_data=(self.X_val, self.y_val) if self.X_val is not None else None,
                                 epochs=self.config.epoch,
                                 callbacks=self.callback_list)
        time.sleep(3)
        gc.collect()

    def simple_model(self):
        print("Creating simple_model ...")
        model = Sequential()
        model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='same',
                         input_shape=(int(self.row * self.config.clip), self.col, self.ch),
                         activation='relu'))
        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.config.label_size))

        model.compile(optimizer="adam", loss="mse")
        print('Model is created and compiled')
        return model

    def get_callback_list(self):
        file_path = self.config.keras_model_path
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False)
        return [checkpoint]

if __name__ == "__main__":
    trainer = CNNTrainer(DataProcessor(DataConfig()))
    trainer.train()





