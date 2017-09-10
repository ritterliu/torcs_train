import os
import socket
import subprocess
import struct
import json

TEST = False

class DriveControl(object):
    '''
    { "steer":0.0000, "accel":0.0, "brake":1.00, "clutch":0.000, "geer":0 }
    '''
    def __init__(self, steer=0.0, accel=0.0, brake=0.0, clutch=0.0, geer=0):
        self.steer = steer
        self.accel = accel
        self.brake = brake
        self.clutch = clutch
        self.geer = geer
    def to_msg(self):
        return '%.4f %.3f %.3f %.3f %d' % (self.steer, self.accel, self.brake, self.clutch, self.geer)


class TorcsEnv(object):
    def __init__(self, torcs_root, track='road/alpine-1', lap=0, with_gt=False):
        if len(torcs_root) == 0 or torcs_root[0] != '/':
            if torcs_root.startswith('./'):
                torcs_root = torcs_root[2:]
            torcs_root = os.getcwd() + '/' + torcs_root
        if not torcs_root.endswith('/'):
            torcs_root += '/'
        self.torcs_root = torcs_root
        self.track = track
        self.lap = lap
        self.with_gt = with_gt
        self.server_addr = ('localhost', 0)
        self.sock = None
        self.child = None
        self.data_sock = None
        self.current_state = None
        self.current_gt = None
        self.start_server()
    def start_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.server_addr)
        self.sock.listen(1)
    def recv_msg(self):
        lenbuff = self.data_sock.recv(4, socket.MSG_WAITALL)
        msg_len, = struct.unpack('=I', lenbuff)
        return self.data_sock.recv(msg_len, socket.MSG_WAITALL)
    def send_msg(self, msg):
        lenbuff = struct.pack('=I', len(msg))
        self.data_sock.sendall(lenbuff)
        self.data_sock.sendall(msg)
    def restart_torcs(self):
        self.close()
        if TEST:
            cwd = self.torcs_root
        else:
            cwd = '%sshare/games/torcs' % self.torcs_root
            print 'cwd:', cwd
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = self.torcs_root + 'lib/torcs/lib:' + env.get('LD_LIBRARY_PATH', '')
        env['XIAOMI_BOT_SERVER'] = '%s:%d' % self.sock.getsockname()
        if self.with_gt:
            env['XIAOMI_BOT_GT'] = 'true'
        if TEST:
            program = '%storcs_test.py' % self.torcs_root
        else:
            program = '%slib/torcs/torcs-bin' % self.torcs_root
        pauto = '-a'
        ptrack = '-b%s' % self.track
        pbot = '-cxiaomi/1'
        pscreen = '-d'
        plap = '-e%d' % self.lap
        self.child = subprocess.Popen([program, pauto, ptrack, pbot, pscreen, plap], cwd=cwd, env=env)
    def close(self):
        if self.data_sock:
            try:
                self.data_sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.data_sock.close()
            self.data_sock = None
        if self.child:
            ret = self.child.wait()
            self.child = None
            if ret != 0:
                raise Exception('child exit with code %d' % ret)
    def recv_state(self):
        msg = self.recv_msg()
        if self.with_gt:
            state, gt = msg.split('\t', 1)
            self.current_state = json.loads(state)
            self.current_gt = json.loads(gt)
        else:
            self.current_state = json.loads(state)
            self.current_gt = None
        return self.current_state
    def reset(self):
        '''
        start a new game, return initial state, like:
           { "time":0.00, "end":false, "distance":-25.01, "damage":0, "lap":0, "gear":0, "speed":0.01, "rpm":94.2, "screen":"/9j/4AAQSkZJRgA...." }
           time: seconds from start
           end: true/false, is game ended
           distance: total distance(m) the car runs
           damage: car damage due to colisions to road/barrier/other car
           lap: current lap
           geer: current geer
           speed: current speed (m/s)
           rpm: current engine rpm
           screen: jpeg image of current driver view
        '''
        self.restart_torcs()
        print('listen on %s:%d, waiting for torcs game to connect...' % self.sock.getsockname())
        self.data_sock, _ = self.sock.accept()
        print('connected!')
        return self.recv_state()
    def step(self, control):
        self.send_msg(control.to_msg())
        state = self.recv_state()
        if state['end']:
            self.close()
        return state


class Robot(object):
    def __init__(self, env):
        self.env = env

    def get_control(self, state):
        gt = self.env.current_gt
        if not gt:
            return DriveControl(accel=0.3, geer=1)
        angle = gt['angle']
        offset = gt['offset']
        angle -= offset / 2
        speed = state['speed']
        return DriveControl(steer=angle / 0.367, accel=0.5 if speed < 30 else 0, geer=1)

    def play(self):
        state = self.env.reset()    
        while not state['end']:
            ctl = self.get_control(state)
            state = self.env.step(ctl)

if __name__ == '__main__':
    env = TorcsEnv(torcs_root='/home/mi/aicontest/aicontest/install/', track='road/alpine-1', lap=1, with_gt=True)
    # env = TorcsEnv(torcs_root='install', track='oval/alpine-2', lap=1, with_gt=True)
    r = Robot(env)
    r.play()