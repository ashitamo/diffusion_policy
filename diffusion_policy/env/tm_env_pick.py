# tm_pick_place_env.py
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from gym import spaces
import gym
import cv2
from diffusion_policy.env.robot import TM5_700, HandEyeCamera



class TMPickPlaceEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    reward_range = (0, 1.0)

    def __init__(self, rate: float = 30.0, gui: bool = False):
        super().__init__()
        print("Init TMPickPlaceEnv with rate:", rate)
        self.rate = rate
        self.np_random = np.random.default_rng(0)
        self._seed = 0
        if gui:
            connection_mode = p.GUI
        else:
            connection_mode = p.DIRECT
        p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(numSolverIterations=150, erp=0.2)
        p.setTimeStep(1.0 / self.rate)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        
        self.render_cache = None

        self.plane_id = p.loadURDF("plane.urdf")
        self.goal_x_min = 0.45
        self.goal_x_max = 0.55
        self.goal_y_min = -0.05
        self.goal_y_max = 0.05
        self._create_goal_zone_visual()
        self.cube_id = None

        self.robot = TM5_700()
        self.camera = HandEyeCamera(
            self.robot.id,
            self.robot.eef_id,
            (0.0074484, -0.067017, 0.0365),
            (0.309, 0, 3.14),
            0.01,
            5.0,
            (320, 240),
            87.0,
        )
        self._init_gui_sliders()

        self.gripper_link_indices = []
        n_joints = p.getNumJoints(self.robot.id)
        for j in range(n_joints):
            info = p.getJointInfo(self.robot.id, j)
            name = info[1].decode("utf-8")
            # 依照 URDF 中命名，自行調整關鍵字即可
            if ("finger" in name) or ("gripper" in name):
                self.gripper_link_indices.append(j)

        self.action_space = spaces.Box(
            low=np.array(
                [0.15, -0.3, 0.0, 0.5*np.pi, -0.5*np.pi, -np.pi, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [0.60, 0.3, 0.55, 1.5*np.pi, 0.5*np.pi, np.pi, 0.085],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.init_obs_space()
        
    def init_obs_space(self):
        ax_low = self.action_space.low
        ax_high = self.action_space.high
        self.x_min = ax_low[0]
        self.x_max = ax_high[0]
        self.x_max = min(self.x_max, self.goal_x_min)
        self.y_min = ax_low[1]
        self.y_max = ax_high[1]

        self.observation_space = spaces.Dict({
            "img": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(96, 96, 3),
                dtype=np.float32
            ),
            "pos_joints": spaces.Box(
                low=np.array(self.robot.arm_lower_limits, dtype=np.float32),
                high=np.array(self.robot.arm_upper_limits, dtype=np.float32),
                shape=(6,),
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
            ),
            "cube_pos": spaces.Box(
                low=np.array([self.x_min, self.y_min, ax_low[2], -np.pi, -np.pi, -np.pi], dtype=np.float32),
                high=np.array([self.x_max, self.y_max, ax_high[2], np.pi, np.pi, np.pi], dtype=np.float32),
                shape=(6,),
                dtype=np.float32
            ),
            "goal_zone": spaces.Box(
                low=np.array([self.x_min, self.x_max, self.y_min, self.y_max], dtype=np.float32),
                high=np.array([self.x_max, self.x_max, self.y_max, self.y_max], dtype=np.float32),
                shape=(4,),
                dtype=np.float32
            ),
        })

    def get_obs(self):
        rgb, depth, seg = self.camera.shot()
        rgb = rgb[:, :, :3].astype(np.uint8)
        self.render_cache = rgb.copy()
        img = rgb[:240,:240,:]
        img = cv2.resize(img, (96, 96))/255.0

        robot_info = self.robot.get_joint_obs()
        pos_joints = np.array(robot_info["pos_joints"], dtype=np.float32)
        pos_ee = np.array(robot_info["pos_ee"], dtype=np.float32)
        gripper_length = np.array([robot_info["gripper_length"]], dtype=np.float32)

        cube_pos, cube_quat = p.getBasePositionAndOrientation(self.cube_id)
        cube_euler = p.getEulerFromQuaternion(cube_quat)
        cube_pos = np.array([*cube_pos, *cube_euler], dtype=np.float32)
        goal_zone = np.array([self.goal_x_min,self.goal_x_max,self.goal_y_min,self.goal_y_max,],dtype=np.float32)

        return {
            "rgb": rgb,
            "img": img,
            "pos_joints": pos_joints,
            "pos_ee": pos_ee,
            "gripper_length": gripper_length,
            "cube_pos": cube_pos,
            "goal_zone": goal_zone,
        }
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _create_goal_zone_visual(self):
        """繪製紅色的 target zone 平面（只是為了讓人看到）"""
        vs = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                (self.goal_x_max - self.goal_x_min) / 2,
                (self.goal_y_max - self.goal_y_min) / 2,
                0.001,
            ],
            rgbaColor=[1, 0, 0, 0.4],
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vs,
            baseCollisionShapeIndex=-1,
            basePosition=[
                (self.goal_x_min + self.goal_x_max) / 2,
                (self.goal_y_min + self.goal_y_max) / 2,
                0.001,
            ],
        )

    def _init_gui_sliders(self):
        self.grip_id = p.addUserDebugParameter("gripper", 0, 0.085, 0.085)
        self.x_id = p.addUserDebugParameter("x", 0.15, 0.6, self.robot.init_pos[0])
        self.y_id = p.addUserDebugParameter("y", -0.4, 0.4, self.robot.init_pos[1])
        self.z_id = p.addUserDebugParameter("z", 0, 0.6, self.robot.init_pos[2])
        self.roll_id = p.addUserDebugParameter("roll", 0.5 * np.pi, 1.5 * np.pi, self.robot.init_pos[3])
        self.pitch_id = p.addUserDebugParameter("pitch", -np.pi, np.pi, self.robot.init_pos[4])
        self.yaw_id = p.addUserDebugParameter("yaw", -np.pi, np.pi, self.robot.init_pos[5])

    def respawn_cube(self):
        """在可達工作空間內隨機生成方塊，並且保證不在 goal zone 裡。"""
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        while True:
            x = self.np_random.uniform(self.x_min, self.x_max)
            y = self.np_random.uniform(self.y_min, self.y_max)
            z = 0.05

            in_goal = (
                self.goal_x_min <= x <= self.goal_x_max and
                self.goal_y_min <= y <= self.goal_y_max
            )
            if not in_goal:
                break   # ok，跳出去

        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[x, y, z],
            useFixedBase=False,
            globalScaling=1.0,
        )
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[0, 1, 0, 1])
        p.changeDynamics(
            self.cube_id,
            -1,
            mass=0.1,
            lateralFriction=1.0,
            rollingFriction=0.01,
            spinningFriction=0.01,
        )


    def reset(self):
        self.robot.reset()
        self.respawn_cube()
        for _ in range(60):
            p.stepSimulation()
        return self.get_obs()

    def step(self, action):
        # gym 會給 numpy array；MultiStepWrapper 也一樣
        x, y, z, roll, pitch, yaw, grip = action.tolist()
        self.robot.move_gripper(float(grip))
        self.robot.move_ee((x, y, z, roll, pitch, yaw), "end")
        p.stepSimulation()

        obs = self.get_obs()
        text = "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            obs["pos_ee"][0], obs["pos_ee"][1], obs["pos_ee"][2], obs["pos_ee"][3], obs["pos_ee"][4], obs["pos_ee"][5], obs["gripper_length"][0]
            )
        try:
            if self.ee_text_id is not None:
                p.removeUserDebugItem(self.ee_text_id)
        except Exception:
            pass
        self.ee_text_id = p.addUserDebugText(
            text,
            textPosition=[0.0, 0.0, 0.8],   # 你可以自己調高低
            textColorRGB=[1, 0, 0],
            textSize=1.4,
            lifeTime=0 
        )
        reward, done = self._get_reward(obs)
        info = {}  # 之後要加 debug 資訊可以放這邊
        return obs, reward, done, info
    
    def _check_cube_in_goal(self, pos, gripper_length):
        x, y, z = pos[:3]
        gr = float(gripper_length[0])
        return (
            self.goal_x_min <= x <= self.goal_x_max
            and self.goal_y_min <= y <= self.goal_y_max
            and z <= 0.04
            and gr >= 0.055   # 這裡是「放開夾爪」的成功條件
        )

    def _check_gripper_contact(self, min_normal_force=1e-4):
        if self.cube_id is None:
            return False
        pts = p.getContactPoints(
            bodyA=self.robot.id,
            bodyB=self.cube_id
        )
        for c in pts:
            linkA = c[3]         # bodyA 的 linkIndex
            normal_force = c[9]  # 法向力
            if linkA in self.gripper_link_indices and normal_force > min_normal_force:
                return True
        return False

    def _compute_goal_proximity_reward(self, cube_pos, is_holding_cube):
        """夾住方塊時，離 goal 越近 reward 越大"""
        if not is_holding_cube:
            return 0.0

        # goal 中心點
        goal_x = 0.5 * (self.goal_x_min + self.goal_x_max)
        goal_y = 0.5 * (self.goal_y_min + self.goal_y_max)
        goal_z = 0.04   # 你可以依需求調整

        goal_center = np.array([goal_x, goal_y, goal_z], dtype=np.float32)
        cube_pos = np.array(cube_pos[:3], dtype=np.float32)

        dist = np.linalg.norm(cube_pos - goal_center)

        # 可以依照 workspace 大小調整這個最大距離
        max_dist = 0.3  # 例：0.3 公尺以外都算「很遠」
        dist_norm = np.clip(dist / max_dist, 0.0, 1.0)

        # 距離越近 reward 越接近 1，越遠越接近 0
        proximity_reward = 1.0 - dist_norm

        # 再乘上一個係數，避免還沒完成就拿到超過 1 的 reward
        proximity_scale = 0.8
        return proximity_scale * proximity_reward


    def _get_reward(self, obs):
        cube_pos = obs["cube_pos"]
        gripper_len = obs["gripper_length"]
        gr = float(gripper_len[0])

        # 原本的 success 判斷（放開夾爪 + cube 在 goal 內）
        done = self._check_cube_in_goal(cube_pos, gripper_len)

        # 有沒有接觸
        has_contact = self._check_gripper_contact()
        # 判斷是否「夾住」方塊：有接觸 + 夾爪收緊
        is_holding_cube = has_contact and (gr < 0.055)

        # 接觸 bonus（碰到就給一點）
        contact_reward = 0.2 if has_contact else 0.0

        # 夾著方塊時，越靠近 goal reward 越高
        proximity_reward = self._compute_goal_proximity_reward(cube_pos, is_holding_cube)

        if done:
            # 完成任務：直接給 1.0
            reward = 1.0
        else:
            # shaping：接觸 + 靠近的分數，加起來最多 < 1
            reward = contact_reward + proximity_reward

        return reward, done

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        if self.render_cache is None:
            self.get_obs()
        rgb = self.render_cache
        return rgb[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect()

    def read_gui_action(self):
        return np.array(
            [
                p.readUserDebugParameter(self.x_id),
                p.readUserDebugParameter(self.y_id),
                p.readUserDebugParameter(self.z_id),
                p.readUserDebugParameter(self.roll_id),
                p.readUserDebugParameter(self.pitch_id),
                p.readUserDebugParameter(self.yaw_id),
                p.readUserDebugParameter(self.grip_id),
            ],
            dtype=np.float32,
        )
