import time
import json
import queue
from pynput import keyboard
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from tm_msgs.srv import SetPositions,SetEvent,SendScript,SetIO
from tm_msgs.msg import FeedbackState
from py_gripper_interfaces.srv import Trajectory
from py_gripper_interfaces.msg import TrajState
from sensor_msgs.msg import Image,PointCloud2
from std_msgs.msg import Float64MultiArray
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

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
        self.color_img_sub = self.create_subscription(Image, '/camera/color/image_raw')
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states')
        
        self.ts = ApproximateTimeSynchronizer(
            [self.color_img_sub, self.pos_sub ],
            queue_size=10,
            slop=0.051  # 1/60+1/30 + 0.001 (30 hz, 60 hz)
        )
        self.ts.registerCallback(self.sync_cam_robot_callback)
        self.obs_queue = queue.Queue(30)
    def sync_cam_robot_callback(self, color_img, pos):
        if self.obs_queue.full():
            self.obs_queue.get()
        self.obs_queue.put((color_img, pos))

    def get_obs(self):
        pass

    def exec_action(self, action):
        pass
       

def main(args=None):
    rclpy.init(args=args)
    tMRealEnv = TMRealEnv()
    rclpy.spin(tMRealEnv)
    tMRealEnv.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()