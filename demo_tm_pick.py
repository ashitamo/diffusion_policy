# demo_tm_pick.py
import numpy as np
import time
import pybullet as p
import click
import cv2
import pygame
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv


def init_joystick():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("[Joystick] 沒有偵測到手把，改用 GUI sliders 控制")
        return None
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"[Joystick] 使用手把：{js.get_name()}")
    return js


@click.command()
@click.option('-o', '--output', default="data/tm_pick_demo.zarr", required=True)
@click.option('-hz', '--control_hz', default=30, type=int)
def main(output, control_hz):
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    env = TMPickPlaceEnv(rate=control_hz, gui=True)
    rate = control_hz
    dt = 1.0 / rate
    joystick = init_joystick()
    print("TM Pick & Place Demo Recorder Started!")
    print("Xbox 手把控制，鍵盤：R 重置、Q 離開、Space 暫停")
    print("----------------------------------------------------------")
    
    while True:
        episode = []

        # seed auto-increment
        seed = replay_buffer.n_episodes
        print(f"\n=== Starting new episode (seed={seed}) ===")

        env.seed(seed)
        obs = env.reset()
        pause = False
        retry = False
        done = False

        ee = obs["pos_ee"]  # 通常是 (x,y,z,roll,pitch,yaw) 或類似
        cmd_x, cmd_y, cmd_z = float(ee[0]), float(ee[1]), float(ee[2])
        cmd_roll = np.pi          # 跟 GUI slider 預設一樣
        cmd_pitch = 0.0
        cmd_yaw = 0.0
        cmd_grip = 0.085          # 夾爪打開

        # 控制速度（你可以依手感調整）
        pos_speed = 0.6     # m/s 位置
        rot_speed = 1.0     # rad/s 轉動
        grip_speed = 0.5   # m/s 夾爪

        t = 0
        while not done:
            time.sleep(dt)

            keys = p.getKeyboardEvents()
            if ord('q') in keys:
                print("Exiting.")
                exit(0)
            if ord('r') in keys:
                print("Retry episode.")
                retry = True
                break
            if 32 in keys:   # Space
                pause = not pause
                time.sleep(0.2)  # prevent bouncing
            if pause:
                continue

            if joystick is not None:
                pygame.event.pump()
                back_btn = joystick.get_button(6)
                start_btn = joystick.get_button(7)

                if back_btn or start_btn:
                    print("[Joystick] RESET episode (BACK/START pressed)")
                    retry = True
                    break   # 跳出 while not done，回到外層重新開始 episode
                # ===============================================

                # 常見的 Xbox 配置（不同手把可能 index 會不一樣）
                lx = joystick.get_axis(0)   # 左搖桿 X
                ly = joystick.get_axis(1)   # 左搖桿 Y
                rx = joystick.get_axis(3)   # 右搖桿 X
                ry = joystick.get_axis(4)   # 右搖桿 Y

                # 位置控制：左搖桿 xy，右搖桿 y 控 z
                cmd_x += lx * pos_speed * dt
                cmd_y += -ly * pos_speed * dt   # y 軸通常反向
                cmd_z += -ry * pos_speed * dt   # push up = z+

                # 旋轉控制：右搖桿 x 控 yaw
                cmd_yaw -= rx * rot_speed * dt

                # LB / RB 控 roll
                LB = joystick.get_button(4)
                RB = joystick.get_button(5)
                if LB:
                    cmd_roll -= rot_speed * dt
                if RB:
                    cmd_roll += rot_speed * dt

                # X / Y 控 pitch
                Xbtn = joystick.get_button(2)
                Ybtn = joystick.get_button(3)
                if Xbtn:
                    cmd_pitch -= rot_speed * dt
                if Ybtn:
                    cmd_pitch += rot_speed * dt

                # A / B 控制夾爪
                A = joystick.get_button(0)
                B = joystick.get_button(1)
                if A:
                    cmd_grip -= grip_speed * dt   # 收夾
                if B:
                    cmd_grip += grip_speed * dt   # 張開
                # clip 到 action_space 範圍
                low = env.action_space.low
                high = env.action_space.high
                cmd = np.array(
                    [cmd_x, cmd_y, cmd_z,
                     cmd_roll, cmd_pitch, cmd_yaw,
                     cmd_grip],
                    dtype=np.float32
                )
                cmd = np.clip(cmd, low, high)
                (cmd_x, cmd_y, cmd_z,
                 cmd_roll, cmd_pitch, cmd_yaw,
                 cmd_grip) = cmd.tolist()
                if cmd[2] <= 0.18:
                    cmd[2] = 0.18
                action = cmd
            else:
                action = env.read_gui_action()

            obs, reward, done, info = env.step(action)

            img = obs["img"]
            pos_joints = obs["pos_joints"]
            pos_ee = obs["pos_ee"]
            cube_pos = obs["cube_pos"]
            goal_zone = obs["goal_zone"]
            gripper_length = obs["gripper_length"]

            episode.append({
                'img': img.astype(np.float32),
                'pos_joints': pos_joints.astype(np.float32),
                'pos_ee': pos_ee.astype(np.float32),
                'cube_pos': cube_pos.astype(np.float32),
                'gripper_length': gripper_length.astype(np.float32),
                'action': np.asarray(action, dtype=np.float32),
            })

            t += 1
            vis = env.render()
            robot_ee = obs["pos_ee"]
            cube_pos = obs["cube_pos"]
            vis = cv2.resize(vis, None, fx=3.0, fy=3.0)
            cv2.putText(vis,
                        f"EE: x={robot_ee[0]:.3f} y={robot_ee[1]:.3f} z={robot_ee[2]:.3f} gripper={gripper_length[0]:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(vis,
                        f"CUBE: x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} z={cube_pos[2]:.3f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 200), 2)
            cv2.putText(vis,
                        f"reward: {reward}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 200), 2)
            cv2.imshow("TM Replay Viewer", vis)
            cv2.waitKey(1)

        if not retry and len(episode) > 1:
            data_dict = {}
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')

            print(f"Episode saved. seed={seed}, length={len(episode)} steps")
        else:
            print(f"Episode discarded. seed={seed}")


if __name__ == "__main__":
    main()
