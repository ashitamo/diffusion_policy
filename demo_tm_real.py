# demo_tm_real.py
import numpy as np
import time
import click
import cv2
import pygame
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.tm_real_env import startEnvNode
import rclpy

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

def read_joystick_action(
    joystick,
    env,
    cmd_x, cmd_y, cmd_z,
    cmd_roll, cmd_pitch, cmd_yaw,
    cmd_grip,
    dt,
    pos_speed,
    rot_speed,
    grip_speed,
):
    """讀取 Xbox 搖桿並更新指令，回傳一個 dict：
    {
        "action": np.ndarray shape (7,),
        "velocity": np.ndarray shape (7,),
        "need_reset": bool,
        "cmd": (cmd_x, cmd_y, cmd_z, cmd_roll, cmd_pitch, cmd_yaw, cmd_grip),
        "record_pressed": bool,   # 這一幀 record 鍵是否有被按住
    }
    """

    # 讓 pygame 更新狀態
    pygame.event.pump()

    # === reset episode: MENU ===
    menu_btn = joystick.get_button(7)
    if menu_btn:
        print("[Joystick] RESET episode (MENU pressed)")
        return {
            "action": None,
            "velocity": None,
            "need_reset": True,
            "cmd": (cmd_x, cmd_y, cmd_z, cmd_roll, cmd_pitch, cmd_yaw, cmd_grip),
            "record_pressed": False,
        }
    # === record 鍵 ===
    record_btn = joystick.get_button(6)
    record_pressed = bool(record_btn)

    # ---- 死區設定 ----
    DEADZONE = 0.1  # 你可以依手感調 0.05 ~ 0.2

    def apply_deadzone(v, dz=DEADZONE):
        if abs(v) < dz:
            return 0.0
        return v

    # 軸讀取（不同手把 index 可能略有不同）
    lx_raw = joystick.get_axis(0)   # 左搖桿 X
    ly_raw = joystick.get_axis(1)   # 左搖桿 Y
    rx_raw = joystick.get_axis(3)   # 右搖桿 X
    ry_raw = joystick.get_axis(4)   # 右搖桿 Y

    # 套用死區
    lx = apply_deadzone(lx_raw)
    ly = apply_deadzone(ly_raw)
    rx = apply_deadzone(rx_raw)
    ry = apply_deadzone(ry_raw)

    # 位置控制：左搖桿 xy，右搖桿 y 控 z
    vx = -lx * pos_speed   # 左搖桿右推 x+
    vy =  ly * pos_speed   # 左搖桿前推 y+
    vz = -ry * pos_speed   # 右搖桿前推 z
    cmd_x += vx * dt
    cmd_y += vy * dt
    cmd_z += vz * dt

    # 旋轉控制：右搖桿 x 控 yaw
    vyaw = -rx * rot_speed
    cmd_yaw += vyaw * dt

    # LB / RB 控 roll
    LB = joystick.get_button(4)
    RB = joystick.get_button(5)
    vroll = 0.0
    if LB:
        vroll = -rot_speed
        cmd_roll += vroll * dt
    if RB:
        vroll = rot_speed
        cmd_roll += vroll * dt

    # X / Y 控 pitch
    Xbtn = joystick.get_button(2)
    Ybtn = joystick.get_button(3)
    vpitch = 0.0
    if Xbtn:
        vpitch = -rot_speed
        cmd_pitch += vpitch * dt
    if Ybtn:
        vpitch = rot_speed
        cmd_pitch += vpitch * dt

    # A / B 控制夾爪
    A = joystick.get_button(0)
    B = joystick.get_button(1)
    vg = 0.0
    if A:
        vg = -grip_speed
        cmd_grip += vg * dt   # 收夾
    if B:
        vg = grip_speed
        cmd_grip += vg * dt   # 張開

    cmd = np.array(
        [cmd_x, cmd_y, cmd_z,
         cmd_roll, cmd_pitch, cmd_yaw,
         cmd_grip],
        dtype=np.float32
    )
    cmd_x, cmd_y, cmd_z, cmd_roll, cmd_pitch, cmd_yaw, cmd_grip = cmd.tolist()

    # 限制邊界
    if cmd_z <= 0.165:
        cmd_z = 0.165
    if cmd_grip <= 0.0:
        cmd_grip = 0.0
    if cmd_grip >= 0.085:
        cmd_grip = 0.085
    
    action = np.array( 
        [cmd_x, cmd_y, cmd_z,
         cmd_roll, cmd_pitch, cmd_yaw,
         cmd_grip],
        dtype=np.float32
    )
    low = env.action_space.low
    high = env.action_space.high
    action = np.clip(action, low, high)
    velocity = np.array(
        [vx, vy, vz,
         vroll, vpitch, vyaw,
         vg],
        dtype=np.float32
    )

    return {
        "action": action,
        "velocity": velocity,
        "need_reset": False,
        "cmd": (cmd_x, cmd_y, cmd_z, cmd_roll, cmd_pitch, cmd_yaw, cmd_grip),
        "record_pressed": record_pressed,
    }


@click.command()
@click.option('-o', '--output', default="data/tm_real_demo.zarr", required=True)
@click.option('-hz', '--control_hz', default=10, type=int)
def main(output, control_hz):
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    env = startEnvNode()
    rate = control_hz
    dt = 1.0 / rate
    joystick = init_joystick()
    if joystick is None:
        print("請使用手把進行操作，程式結束。")
        return
    print("TM Pick & Place Demo Recorder Started!")
    print("Xbox 手把控制，鍵盤：R 重置、Q 離開、Space 暫停")
    print("----------------------------------------------------------")
    
    while True:
        episode = []
        # seed auto-increment
        seed = replay_buffer.n_episodes
        while not env.is_ready():
            print("waiting for environment...")
            time.sleep(0.1)

       

        # env.seed(seed)
        obs = env.reset()
        retry = False
        done = False
        is_recording = True      # 一開始還不錄
        prev_record_pressed = False

        print(f"\n=== Starting new episode (seed={seed}) ===")
        obs = env.get_obs()
        ee = obs["pos_ee"]  # (x,y,z,roll,pitch,yaw)
        cmd_x, cmd_y, cmd_z = float(ee[0]), float(ee[1]), float(ee[2])
        cmd_roll = float(ee[3])  # 跟 GUI slider 預設一樣
        cmd_pitch = float(ee[4])
        cmd_yaw = float(ee[5])
        cmd_grip = float(obs["gripper_length"])          # 夾爪打開

        # 控制速度（你可以依手感調整）
        pos_speed = 0.1    # m/s 位置
        rot_speed = 0.5    # rad/s 轉動
        grip_speed = 0.06    # m/s 夾爪

        t = 0
        while not done:
            time.sleep(dt)
            obs = env.get_obs()
            img = obs["img"]
            pos_ee = obs["pos_ee"]
            gripper_length = obs["gripper_length"]

            joyDict = read_joystick_action(
                joystick, env,
                cmd_x, cmd_y, cmd_z,
                cmd_roll, cmd_pitch, cmd_yaw,
                cmd_grip,
                dt,
                pos_speed,
                rot_speed,
                grip_speed,
            )
            action = joyDict["action"]
            velocity = joyDict["velocity"]
            need_reset = joyDict["need_reset"]
            (cmd_x, cmd_y, cmd_z, cmd_roll, cmd_pitch, cmd_yaw, cmd_grip) = joyDict["cmd"]
            
            record_pressed = joyDict["record_pressed"]
            # 邊緣觸發：這一幀從「沒按」→「按下去」
            if record_pressed and not prev_record_pressed:
                if not is_recording:
                    # 第一次按：開始錄
                    is_recording = True
                    print("Recording started.")
                else:
                    # 第二次按：結束錄影，並結束這個 episode
                    print("Recording stopped, ending episode.")
                    done = True
                    break
            prev_record_pressed = record_pressed

            if need_reset:
                retry = True
                break

            rgb = obs["rgb"]
            # show big text pos ee
            cv2.putText(
                rgb,
                f"EE Pos: x={pos_ee[0]:.3f} y={pos_ee[1]:.3f} z={pos_ee[2]:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
            # show gripper length
            cv2.putText(
                rgb,
                f"Gripper Length: {gripper_length[0]:.3f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )
            # show step 
            cv2.putText(
                rgb,
                f"Step: {t}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
            cv2.imshow("TM Replay Viewer", rgb)
            cv2.waitKey(1)
            env.exec_action(action, velocity, duration=dt)
            # print(f"Recording {is_recording} Step {t}")
            if is_recording:
                episode.append({
                    'img': img.astype(np.float32),
                    'pos_ee': pos_ee.astype(np.float32),
                    'gripper_length': gripper_length.astype(np.float32),
                    'action': np.asarray(action, dtype=np.float32),
                })
                t += 1

        if not retry and len(episode) > 1:
            data_dict = {}
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f"== Episode saved. seed={seed}, length={len(episode)} steps ==\n")
        else:
            print(f"== Episode discarded. seed={seed} ==\n")


if __name__ == "__main__":
    main()
    rclpy.shutdown()
