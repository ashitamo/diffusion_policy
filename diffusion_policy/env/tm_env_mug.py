# tm_env.py
import numpy as np
import pybullet as p
import pybullet_data

from diffusion_policy.env.robot import TM5_700, HandEyeCamera
from scipy.spatial.transform import Rotation as R

class TM5DPEnv:
    """
    給 diffusion policy 用的 TM5 + 手眼相機 + 單一物體的小環境
    obs:
        - rgb: (H, W, 4) uint8（pybullet 會回傳 RGBA）
        - robot_q: (n_joints,) float32
        - robot_ee: (3,) float32
    action: np.array(7)
        [x, y, z, roll, pitch, yaw, gripper_opening]
    """

    def __init__(self, rate: float = 240.0):
        # ----------------------------
        # 連線 & 物理設定
        # ----------------------------
        self.rate = rate
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            erp=0.2,
        )
        p.setTimeStep(1.0 / self.rate)

        # ----------------------------
        # 地板 & 物體
        # ----------------------------
        self.plane_id = p.loadURDF("plane.urdf")
        self.obj_id = p.loadURDF(
            "urdf/object_demo.urdf",
            [0.3, 0.0, 0.1],
            R.from_euler("xyz", [1.57, 0, 0]).as_quat(),
            globalScaling=0.8,
        )
        # 設定物體摩擦
        p.changeDynamics(
            self.obj_id, -1,
            lateralFriction=1.5,
            spinningFriction=0.5,
            rollingFriction=0.0005,
            frictionAnchor=1,
        )

        # ----------------------------
        # 機械臂 & 相機
        # ----------------------------
        self.robot = TM5_700()
        self.camera = HandEyeCamera(
            self.robot.id,
            self.robot.eef_id,
            (-0.0074484, 0.067017, 0.0365),
            (0.209, 0.0, 0.0),
            0.1,
            5.0,
            (848, 480),
            80.0,
        )

        # ----------------------------
        # GUI sliders（方便手動玩 / 收集 demo）
        # ----------------------------
        self.gripper_opening_length = p.addUserDebugParameter(
            "gripper_opening_length", 0, 0.085, 0.085
        )
        self.xid = p.addUserDebugParameter("x", 0.15, 0.6, 0.3)
        self.yid = p.addUserDebugParameter("y", -0.4, 0.4, 0.0)
        self.zid = p.addUserDebugParameter("z", 0.0, 0.6, 0.3)
        self.rollId = p.addUserDebugParameter("roll", 0.5*np.pi, 1.5*np.pi, 3.14)
        self.pitchId = p.addUserDebugParameter("pitch", -np.pi, np.pi, 0.0)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi, np.pi, 0.0)

    # ----------------------------
    # gym style API
    # ----------------------------
    def reset(self):
        # reset robot
        self.robot.reset()
        # reset object
        p.resetBasePositionAndOrientation(
            self.obj_id,
            [0.3, 0.0, 0.1],
            R.from_euler("xyz", [1.57, 0, 0]).as_quat(),
        )
        # 幾步讓東西穩定
        for _ in range(10):
            p.stepSimulation()

        return self.get_obs()

    def step(self, action: np.ndarray):
        """
        action: [x, y, z, roll, pitch, yaw, gripper_opening]
        """
        assert action.shape[-1] == 7
        x, y, z, roll, pitch, yaw, grip = action.tolist()

        # 控制
        self.robot.move_gripper(float(grip))
        self.robot.move_ee((x, y, z, roll, pitch, yaw), 'end')

        # 模擬一步
        p.stepSimulation()

        obs = self.get_obs()
        reward = 0.0
        done = False
        info = {}

        return obs, reward, done, info

    def get_obs(self):
        rgb, depth, seg = self.camera.shot()
        j = self.robot.get_joint_obs()

        obs = {
            "rgb": rgb[:, :, :3].astype(np.uint8),
            "robot_q": np.array(j["positions"], dtype=np.float32),
            "robot_ee": np.array(j["ee_pos"], dtype=np.float32),
        }
        return obs

    def render(self):
        rgb, _, _ = self.camera.shot()
        return rgb

    # 小工具：從 slider 讀取當前 GUI 目標
    def read_gui_action(self):
        grip_state = p.readUserDebugParameter(self.gripper_opening_length)
        x = p.readUserDebugParameter(self.xid)
        y = p.readUserDebugParameter(self.yid)
        z = p.readUserDebugParameter(self.zid)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        return np.array([x, y, z, roll, pitch, yaw, grip_state], dtype=np.float32)
