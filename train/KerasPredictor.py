# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from Config import *
from DataProcessor import DataProcessor


class CNNPredictor:
    def __init__(self, model_path=None):
        self.config = DataConfig()
        self.dp = DataProcessor(self.config)
        self.num_channels = self.config.num_channels
        self.row = self.config.img_height
        self.col = self.config.img_width
        self.ch = self.config.num_channels
        self.model = self.load_model(model_path)

    def load_model(self, model_path=None):
        return load_model(model_path)

    def predict(self, img_json):
        X = np.zeros((1, int(self.row * self.config.clip), self.col, self.ch), dtype=np.float32)
        X[0] = self.dp.get_X(img_json)
        return self.model.predict(X)[0]
