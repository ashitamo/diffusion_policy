import time
import json
import queue
import cv2
from pynput import keyboard
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tm_msgs.msg import FeedbackState
from robotiq_85_msgs.msg import GripperStat
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
import threading
from gym import spaces
import gym
from py_gripper_interfaces.srv import Trajectory

W = set(['w'])
S = set(['s'])
A = set(['a'])
D = set(['d'])
WD = set(['w','d'])
WA = set(['w','a'])
SD = set(['s','d'])
SA = set(['s','a'])
UP = set(['p'])
DOWN = set([';'])
LEFT = set([keyboard.Key.left])
RIGHT = set([keyboard.Key.right])

class TMRealEnv(Node):
    def __init__(self):
        super().__init__('tm_real_env')

        self.init_space()
        self.init_pos = np.array([0.0, -0.4, 0.5, 3.14159, 0.0, 3.14159], dtype=np.float32)


        self.cv_bridge = CvBridge()
        self.color_img_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.pos_sub = Subscriber(self, FeedbackState, '/feedback_states')
        self.gripper_cmd_sub = Subscriber(self, GripperStat, '/gripper/stat')
        self.action_pub = self.create_publisher(Float64MultiArray, '/joy', 30)
        self.action_client = self.create_client(Trajectory, '/trajectory')
        while not self.action_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.ts = ApproximateTimeSynchronizer(
            [self.color_img_sub, self.pos_sub, self.gripper_cmd_sub],
            queue_size=10,
            slop=0.051  # 1/60+1/30 + 0.001 (30 hz, 60 hz)
        )
        self.ts.registerCallback(self.sync_cam_robot_callback)
        self.obs_queue = queue.Queue(30)
        self.get_logger().info('TM Real Env initialized.')

    def init_space(self):
        self.action_space = spaces.Box(
            low=np.array(
                [-0.5, -0.6, 0.0, 0.5*np.pi, -0.5*np.pi, -np.pi, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [0.50, -0.2, 0.6, 1.5*np.pi, 0.5*np.pi, 2*np.pi, 0.085],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "img": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(256, 256, 3),
                dtype=np.float32
            ),
            "pos_ee": spaces.Box(
                low=self.action_space.low[:6],
                high=self.action_space.high[:6],
                shape=(6,),
                dtype=np.float32
            ),
            "gripper_length": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([0.085], dtype=np.float32),
                shape=(1,),
                dtype=np.float32
            )
        })

    def sync_cam_robot_callback(self, color_img, pos, gripper_stat):
        if self.obs_queue.full():
            self.obs_queue.get()
        self.obs_queue.put((color_img, pos, gripper_stat))
        # self.get_logger().info('img timestamp: %.6f' % color_img.header.stamp.sec + '%.6f' % (color_img.header.stamp.nanosec*1e-9))
        # self.get_logger().info('pos timestamp: %.6f' % pos.header.stamp.sec + '%.6f' % (pos.header.stamp.nanosec*1e-9))
        # self.get_logger().info('gripper timestamp: %.6f' % gripper_stat.header.stamp.sec + '%.6f' % (gripper_stat.header.stamp.nanosec*1e-9))
    def is_ready(self):
        return not self.obs_queue.empty()

    def reset(self):
        # move to init pos
        action = np.concatenate([self.init_pos, [0.085]])
        velocity = np.ones(7, dtype=np.float32) * 0.1
        self.exec_action(action, velocity, duration=2.0)

        idx = 0
        while True:
            time.sleep(0.1)
            self.get_logger().info('Waiting for robot to reach initial position...')
            obs = self.get_obs()
            if np.linalg.norm(obs['pos_ee'] - self.init_pos) < 5e-3:
                break
            if idx >= 50:
                break
            idx += 1
        obs = self.get_obs()
        return obs

    def get_obs(self):
        # obs = {
        #     'img': None,
        #     'pos_ee': None,
        #     'gripper_length': None
        # }
        obs = {}
        if not self.obs_queue.empty():
            color_img, pos, gripper_stat = self.obs_queue.queue[-1]
            img = self.cv_bridge.imgmsg_to_cv2(color_img, desired_encoding='bgr8')
            pos_ee = np.array(pos.tool_pose, dtype=np.float32)
            gripper_length = np.array([gripper_stat.position], dtype=np.float32)
            obs['rgb'] = img.copy()
            #crop to square
            h, w, _ = img.shape
            min_dim = min(h, w)
            img = img[(h - min_dim) // 2:(h + min_dim) // 2, (w - min_dim) // 2:(w + min_dim) // 2]
            # print('Original image shape:', img.shape)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.moveaxis(img, -1, 0)  # HWC to CHW
            obs['img'] = img
            obs['pos_ee'] = pos_ee
            obs['gripper_length'] = gripper_length
        return obs

    def exec_action(self, action, velocity, duration=0.1):
        # position = action[:6].tolist()
        # velocity = velocity[:6].tolist()
        # gripper_length = float(action[-1])
        # self.req = Trajectory.Request()
        # self.req.mode = Trajectory.Request.PATH  # position control
        # self.req.duration = duration
        # self.req.velocity = velocity
        # self.req.positions = position
        # self.req.grip = gripper_length
        # future = self.action_client.call_async(self.req)

        # if np.max(np.abs(velocity)) >= 1e-3:
        #     msg = Float64MultiArray()
        #     msg.data = action.tolist()
        #     self.action_pub.publish(msg)

        position = action[:6].tolist()
        velocity = velocity.tolist()
        gripper_length = float(action[-1])
        self.req = Trajectory.Request()
        self.req.mode = Trajectory.Request.JOY
        self.req.duration = duration
        self.req.velocity = velocity
        self.req.positions = position
        self.req.grip = float(gripper_length)
        future = self.action_client.call_async(self.req)

def startEnvNode():
    rclpy.init()
    tMRealEnv = TMRealEnv()
    executor = MultiThreadedExecutor()
    executor.add_node(tMRealEnv)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return tMRealEnv

def main(args=None):
    tMRealEnv = startEnvNode()
    try:
        while rclpy.ok():
            time.sleep(0.1)
            print('Getting observation...')
            obs = tMRealEnv.get_obs()
            if obs:
                print('Observation obtained:')
            time.sleep(0.1)
            print('Getting observation...')
            obs = tMRealEnv.get_obs()
            if obs:
                print('Observation obtained:')
                print('Image shape:', obs['img'].shape)
                print('End-effector position:', obs['pos_ee'])
                print('Gripper length:', obs['gripper_length'])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()