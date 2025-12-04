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
    """
    TM5 Pick & Place 任務環境：
    - 一顆綠色方塊會隨機生成於一定 x,y 範圍
    - 右側地面有一塊紅色的目標區域
    - 若方塊的中心掉入目標區域 -> done=True 並 reward=+1
    """

    # ⭐ 給 Gym / VectorEnv / VideoWrapper 用的標準欄位
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

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

        self.plane_id = p.loadURDF("plane.urdf")
        self.goal_x_min = 0.4
        self.goal_x_max = 0.6
        self.goal_y_min = -0.1
        self.goal_y_max = 0.1
        self._create_goal_zone_visual()
        self.robot = TM5_700()
        self.camera = HandEyeCamera(
            self.robot.id,
            self.robot.eef_id,
            (0.0074484, -0.067017, 0.0365),
            (0.309, 0, 3.14),
            0.01,
            5.0,
            (848, 480),
            87.0,
        )
        self.gripper_link_indices = []
        n_joints = p.getNumJoints(self.robot.id)
        for j in range(n_joints):
            info = p.getJointInfo(self.robot.id, j)
            name = info[1].decode("utf-8")
            # 依照 URDF 中命名，自行調整關鍵字即可
            if ("finger" in name) or ("gripper" in name):
                self.gripper_link_indices.append(j)
        print("[TMPickPlaceEnv] gripper links:", self.gripper_link_indices)
        self._init_gui_sliders()
        self.cube_id = None

        self.action_space = spaces.Box(
            low=np.array(
                [0.15, -0.4, 0.0, 0.5 * np.pi, -np.pi, -np.pi, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [0.60, 0.4, 0.6, 1.5 * np.pi, np.pi, np.pi, 0.085],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        first_obs = self.reset()
        self.observation_space = spaces.Dict({
             "img": spaces.Box(
                low=0.0,
                high=1.0,
                shape=first_obs["img"].shape,   # (3,96,96)
                dtype=np.float32
            ),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=first_obs["state"].shape,   # (16,)
                dtype=np.float32
            ),
        })
        self.render_cache = None

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

    def respawn_cube(self, margin=0.02):
        """在可達工作空間內隨機生成方塊，並且保證不在 goal zone 裡。"""

        # 1) 先把舊方塊刪掉
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        # 2) 從「手臂的 action_space」裡取一個位置，
        #    確保方塊一定在手臂能到的範圍內
        ax_low = self.action_space.low
        ax_high = self.action_space.high

        # 取出 x, y 的可達範圍，並預留一點 margin
        x_min = ax_low[0] + margin
        x_max = ax_high[0] - margin
        y_min = ax_low[1] + margin
        y_max = ax_high[1] - margin

        # 如果你根本就只想讓方塊出現在「左半邊（非 goal 一側）」，
        # 可以把 x_max 硬切在 goal_x_min 左邊一點
        x_max = min(x_max, self.goal_x_min - margin)

        # 3) 拒絕取樣：如果剛好 sample 到 goal zone，就重抽
        while True:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            z = 0.05

            # 檢查是不是落在 goal zone（只看 x,y 就好）
            in_goal = (
                self.goal_x_min <= x <= self.goal_x_max and
                self.goal_y_min <= y <= self.goal_y_max
            )
            if not in_goal:
                break   # ok，跳出去

        # 4) 生成方塊
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[x, y, z],
            useFixedBase=False,
            globalScaling=1.0,
        )
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[0, 1, 0, 1])


    def reset(self):
        self.robot.reset()
        self.respawn_cube()
        for _ in range(50):
            p.stepSimulation()
        return self.get_obs()

    def step(self, action):
        # gym 會給 numpy array；MultiStepWrapper 也一樣
        x, y, z, roll, pitch, yaw, grip = action.tolist()
        self.robot.move_gripper(float(grip))
        self.robot.move_ee((x, y, z, roll, pitch, yaw), "end")
        p.stepSimulation()

        obs = self.get_obs()
        ee_x, ee_y, ee_z = obs["robot_ee"][:3]
        text = f"EE pos: x={ee_x:.3f}, y={ee_y:.3f}, z={ee_z:.3f}"
        try:
            # 刪掉上一個文字（避免越畫越多）
            if self.ee_text_id is not None:
                p.removeUserDebugItem(self.ee_text_id)
        except Exception:
            pass
        self.ee_text_id = p.addUserDebugText(
            text,
            textPosition=[0.0, 0.0, 0.8],   # 你可以自己調高低
            textColorRGB=[1, 0, 0],
            textSize=1.4,
            lifeTime=0  # 0 = 永久顯示，直到你手動 remove / 更新
        )
        
        # reward：方塊進入目標區域
        reward, done = self._get_reward(obs)

        info = {}  # 之後要加 debug 資訊可以放這邊
        return obs, reward, done, info
    
    def _check_cube_in_goal(self, pos, gripper_length):
        x, y, z = pos
        gr = float(gripper_length[0])
        return (
            self.goal_x_min <= x <= self.goal_x_max
            and self.goal_y_min <= y <= self.goal_y_max
            and z <= 0.04
            and gr >= 0.055
        )
    
    def _check_gripper_contact(self, min_normal_force=1e-4):
        """
        回傳 True/False：夾爪有沒有碰到方塊。
        只看 gripper_link_indices 裡的 link。
        """
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


    def _get_reward(self, obs):
        cube_pos = obs["cube_pos"]
        gripper_len = obs["gripper_length"]

        # 原本的 success 判斷
        done = self._check_cube_in_goal(cube_pos, gripper_len)

        # 1) shaping: 夾爪有碰到方塊就給一點獎勵
        has_contact = self._check_gripper_contact()
        contact_reward = 0.2 if has_contact else 0.0

        # 2) 完成任務就直接給 1.0 蓋過去
        if done:
            reward = 1.0
        else:
            reward = contact_reward

        return reward, done

    def get_obs(self):
        rgb, depth, seg = self.camera.shot()
        rgb = rgb[:, :, :3].astype(np.uint8)
        self.render_cache = rgb
        j = self.robot.get_joint_obs()
        robot_q = np.array(j["positions"], dtype=np.float32)
        robot_ee = np.array(j["ee_pos"], dtype=np.float32)[:6]
        gripper_length = self.robot.get_gripper_length()
        gripper_length = np.array([gripper_length], dtype=np.float32)

        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos_arr = np.array(cube_pos, dtype=np.float32)
        goal_zone = np.array([self.goal_x_min,self.goal_x_max,self.goal_y_min,self.goal_y_max,],dtype=np.float32)

        rgb_small = cv2.resize(rgb, (96, 96))
        image = np.moveaxis(rgb_small.astype(np.float32) / 255.0, -1, 0)

        state = np.concatenate(
            [robot_q, robot_ee, gripper_length, cube_pos_arr,goal_zone],
            axis=0
        ).astype(np.float32)

        return {
            # 原本就有的（方便 debug / demo 用）
            "rgb": rgb,
            "robot_q": robot_q,
            "robot_ee": robot_ee,
            "cube_pos": cube_pos_arr,
            "gripper_length": gripper_length,
            "goal_zone": goal_zone,
            "img": image,   # (3,96,96) float32 [0,1]
            "state": state,   # (16,) float32
        }
    # ----------------------------------------------------------
    # Gym 期待的 render() 介面
    # ----------------------------------------------------------
    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        if self.render_cache is None:
            self.get_obs()
        rgb = self.render_cache
        return rgb[:, :, :3].astype(np.uint8)

    def close(self):
        # 如果之後有開多個 client，可以在這邊做 p.disconnect()
        pass

    # ----------------------------------------------------------
    # 從 GUI slider 讀 action（手動控制）
    # ----------------------------------------------------------
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
