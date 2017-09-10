# -*- coding: utf-8 -*-
from KerasPredictor import CNNPredictor
from TFTrainerPredictor import nvidia_net
from Config import *


class DriveControl(object):
    # { "steer":0.0000, "accel":0.0, "brake":1.00, "clutch":0.000, "geer":0 }
    def __init__(self, steer=0.0, accel=0.0, brake=0.0, clutch=0.0, geer=0):
        self.steer = steer
        self.accel = accel
        self.brake = brake
        self.clutch = clutch
        self.geer = geer

    def to_msg(self):
        return '%.4f %.3f %.3f %.3f %d' % (self.steer, self.accel, self.brake, self.clutch, self.geer)

    def to_string(self):
        return 'steer:%.4f accel:%.3f brake:%.3f clutch:%.3f geer:%d' % (self.steer, self.accel, self.brake, self.clutch, self.geer)


class Driver(object):
    def __init__(self, model_path=None):
        #self.predictor = CNNPredictor(model_path)
        self.predictor = nvidia_net() if DataConfig.model_type == 'tensorflow' else CNNPredictor(model_path)

    def get_gear(self, speed):
        assert speed >= 0
        up_count = [0, 21, 33, 45, 57, 69]
        for i in reversed(range(6)):
            if speed >= up_count[i]:
                return i+1

    def drive(self, state):
        steer = self.predictor.predict({'state': state})[1]
        now_speed = state['speed']
        accelerate = 0.8 if now_speed < 20 else 0
        print now_speed
        return DriveControl(steer=steer, accel=accelerate, brake=0, geer=self.get_gear(now_speed))
