import pybullet as p
import math
from collections import namedtuple
import numpy as np

class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            print(info)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)


class Panda(RobotBase):
    def __init_robot__(self):
        # define the robot
        # see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF('./urdf/panda.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.04]
        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.id,
                               9,
                               self.id,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [9, 10]:
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, open_length, force=20)


class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)


class UR5Robotiq140(UR5Robotiq85):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
        # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.

    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

class TM5_700(object):
    def __init__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [0.42027483543494104, -0.4468463190679861, 1.3351729563175363, -0.09148347708718872, 1.5328229089147611, 1.6689732478233352]
        self.id = p.loadURDF(
            './urdf/tm5-700-nominal.urdf',
            useFixedBase=True, 
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self.gripper_range = [0, 0.085]
        self.__parse_joint_info__()
        self.__post_load__()
        self.init_pos = [0.3, 0.0, 0.3,3.14,0.0,0.0]
    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)
            # print(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance
        
        p.changeDynamics(
            bodyUniqueId=self.id,
            linkIndex=11,
            lateralFriction=4.0,      # 接觸面摩擦係數，預設大概 0.5
            spinningFriction=2.0,     # 旋轉摩擦，防止物體在指尖上旋轉
            rollingFriction=2.0,
            frictionAnchor=1          # 啟用 friction anchor，抓取比較穩
        )

        p.changeDynamics(
            bodyUniqueId=self.id,
            linkIndex=12,
            lateralFriction=4.0,      # 接觸面摩擦係數，預設大概 0.5
            spinningFriction=2.0,     # 旋轉摩擦，防止物體在指尖上旋轉
            rollingFriction=2.0,
            frictionAnchor=1          # 啟用 friction anchor，抓取比較穩
        )
        p.changeDynamics(
            bodyUniqueId=self.id,
            linkIndex=17,
            lateralFriction=4.0,      # 接觸面摩擦係數，預設大概 0.5
            spinningFriction=2.0,     # 旋轉摩擦，防止物體在指尖上旋轉
            rollingFriction=2.0,
            frictionAnchor=1          # 啟用 friction anchor，抓取比較穩
        )
        p.changeDynamics(
            bodyUniqueId=self.id,
            linkIndex=16,
            lateralFriction=4.0,      # 接觸面摩擦係數，預設大概 0.5
            spinningFriction=2.0,     # 旋轉摩擦，防止物體在指尖上旋轉
            rollingFriction=2.0,
            frictionAnchor=1          # 啟用 friction anchor，抓取比較穩
        )
    def reset(self):
        self.move_ee(self.init_pos, 'end')
        self.move_gripper(0.085)
        
    def get_gripper_length(self):
        length = 0.1143 * math.sin(0.715 - p.getJointState(self.id, self.mimic_parent_id)[0])
        return length

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            # current_q = [
            #     p.getJointState(self.id, jid)[0]
            #     for jid in self.arm_controllable_joints
            # ]
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(
                self.id,
                self.eef_id,
                pos,
                orn,
                self.arm_lower_limits,
                self.arm_upper_limits,
                self.arm_joint_ranges,
                self.arm_rest_poses,
                # [0.01,  0.01, 0.01, 0.01, 0.01, 0.01,
                #  0.01, 0.01, 0.01, 0.01, 0.01, 0.01],            # jointDamping 正確位置
                maxNumIterations=200        # 這個也不能放在最後，只能拆開寫
            )

        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)
        # print("action: ", np.array2string(np.array(action), separator=', ',precision=3))
    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        temp = p.getLinkState(self.id, self.eef_id)
        ee_pos,ee_quat = temp[0], temp[1]
        ee_euler = p.getEulerFromQuaternion(ee_quat)
        gripper_length = self.get_gripper_length()
        ee_pos = np.concatenate((ee_pos, ee_euler))
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos, gripper_length=gripper_length)
class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self, need_seg=False):
        flags = p.ER_NO_SEGMENTATION_MASK if not need_seg \
                else p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.projection_matrix,
            shadow=0,
            flags=flags,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,  # GUI 時用這個就好                                           
        )
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        # reference: https://stackoverflow.com/a/62247245
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)

class HandEyeCamera:
    def __init__(self,
                 robot_id,
                 link_id,
                 cam_offset_pos,      # 相機在 link 座標系下的位置 [x,y,z]
                 cam_offset_rpy,      # 相機在 link 座標系下的 rpy
                 near, far,
                 size, fov):
        self.robot_id = robot_id
        self.link_id = link_id

        self.offset_pos = np.array(cam_offset_pos, dtype=float)
        self.offset_rpy = np.array(cam_offset_rpy, dtype=float)

        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, aspect, self.near, self.far
        )

        # 初始先隨便設一個 view_matrix，之後每次 shot() 會更新
        self.view_matrix = np.eye(4, dtype=float).reshape(-1).tolist()
        self._update_tran_pix_world()

        # 定義相機在自己座標系下的 forward / up 方向
        # 這裡假設相機 local z 朝前，y 朝上
        self.local_forward = np.array([0, 0, 1], dtype=float)
        self.local_up = np.array([0, 1, 0], dtype=float)

        # 相機看出去多遠當作 target（可以自己調）
        self.target_dist = 1.0

    def _update_tran_pix_world(self):
        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def _update_view_from_robot(self):
        # 1. 取得手臂上某個 link 的世界位姿
        link_state = p.getLinkState(self.robot_id, self.link_id)
        ee_pos = np.array(link_state[0])
        ee_orn = link_state[1]

        # 2. 把 quaternion 轉成旋轉矩陣
        R_ee = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)

        # 3. 相機相對這個 link 的旋轉
        offset_q = p.getQuaternionFromEuler(self.offset_rpy)
        R_offset = np.array(p.getMatrixFromQuaternion(offset_q)).reshape(3, 3)

        # 4. 相機在世界座標系下的位置 / 旋轉
        cam_pos = ee_pos + R_ee @ self.offset_pos
        R_cam = R_ee @ R_offset

        # 5. 算出世界座標下的 forward / up / target
        forward_world = R_cam @ self.local_forward
        up_world = R_cam @ self.local_up
        cam_target = cam_pos + forward_world * self.target_dist

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=up_world.tolist()
        )

        # 6. 因為 view_matrix 改了，要重新算 pixel->world transform
        self._update_tran_pix_world()

    def shot(self, need_seg=False):
        # ⭐ 每次拍照前，先用當前手臂姿態更新相機 pose
        self._update_view_from_robot()

        flags = p.ER_NO_SEGMENTATION_MASK if not need_seg \
            else p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

        _, _, rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.projection_matrix,
            shadow=0,
            flags=flags,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgb, depth, seg

    def rgbd_2_world(self, w, h, d):
        # 跟你原本一樣，只是 tran_pix_world 會一直更新
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1.0))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]
        return position[:3]

    def rgbd_2_world_batch(self, depth):
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array(
            [x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]
        ).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
