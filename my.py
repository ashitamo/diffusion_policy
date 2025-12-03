import os
import time
import math
from collections import namedtuple

import numpy as np
import pybullet as p
import pybullet_data
from attrdict import AttrDict
from diffusion_policy.env.robot import TM5_700,Camera,HandEyeCamera
import cv2


# ----------------------------
# 基本設定
# ----------------------------
SERVER_MODE = p.GUI
physicsClient = p.connect(SERVER_MODE)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setPhysicsEngineParameter(
    fixedTimeStep=1.0 / 240.0,
    numSolverIterations=150,
    erp=0.2
)

# ----------------------------
# 地板
# ----------------------------
plane_id = p.loadURDF("plane.urdf")
obj_id = p.loadURDF("urdf/object_demo.urdf", [0.3, 0.0, 0.2],globalScaling=0.8)
p.changeDynamics(obj_id, -1,
    lateralFriction=1.5,
    spinningFriction=0.5,
    rollingFriction=0.0005,
    frictionAnchor=1
)

# ----------------------------
# 載入 TM5-700 機械臂
# ----------------------------
robot = TM5_700()
# tm_id = p.loadURDF(
#     "urdf/tm5-700-nominal.urdf",
#     basePosition=[0, 0, 0],
#     useFixedBase=True,
# )
camera = HandEyeCamera(
    robot.id,
    robot.eef_id,
    (-0.0074484, 0.067017, 0.0365),
    (0.209, 0, 0),
    0.1,
    5,
    (848, 480),
    80    
)
gripper_opening_length = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.085)
xid = p.addUserDebugParameter("x", 0.15, 0.6, 0.3)
yid = p.addUserDebugParameter("y", -0.4, 0.4, 0)
zid = p.addUserDebugParameter("z", 0, 0.6, 0.3)
rollId = p.addUserDebugParameter("roll", 0.5*np.pi, 1.5*np.pi, 3.14)
pitchId = p.addUserDebugParameter("pitch", -np.pi, np.pi, 0.0)
yawId = p.addUserDebugParameter("yaw", -np.pi, np.pi, 0.0)
# ----------------------------
# 主迴圈：只控制夾爪開合
# ----------------------------
t = 0
rate = 240.0
p.setTimeStep(1./rate) # 240 Hz simulation
fps = 30
while True:
    time.sleep(1.0 / rate)
    grip_state = p.readUserDebugParameter(gripper_opening_length)
    x = p.readUserDebugParameter(xid)
    y = p.readUserDebugParameter(yid)
    z = p.readUserDebugParameter(zid)
    roll = p.readUserDebugParameter(rollId)
    pitch = p.readUserDebugParameter(pitchId)
    yaw = p.readUserDebugParameter(yawId)
    robot.move_gripper(float(grip_state))
    robot.move_ee((x, y, z, roll, pitch, yaw), 'end')
    # robot.move_ee(robot.arm_rest_poses, 'end')
    if t % (int(1/fps*rate)) == 0:
        rgb, depth, segm = camera.shot()

    t+=1
    # print(robot.get_joint_obs()['positions'])
    p.stepSimulation()
