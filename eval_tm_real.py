# eval_tm_real.py
import time
import click
import cv2
import pygame
import numpy as np
import torch
import hydra
import dill
import rclpy

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.real_world.tm_real_env import startEnvNode

########################################
# 搖桿初始化 & 讀取函式（跟 demo_tm_real 類似）
########################################

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ 沒有偵測到搖桿，請插上 Xbox 手把！")
        return None
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"🎮 使用手把：{js.get_name()}")
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
    """
    跟你 demo_tm_real 的版本一樣：
    從搖桿讀取 → 更新指令 → 回傳 action / velocity / 是否 reset / 是否按下 record。
    """
    pygame.event.pump()

    # MENU / START 按鈕 → reset episode
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

    # Record / BACK 鍵
    record_btn = joystick.get_button(6)
    record_pressed = bool(record_btn)

    # 死區
    DEADZONE = 0.1

    def apply_deadzone(v, dz=DEADZONE):
        if abs(v) < dz:
            return 0.0
        return v

    lx = apply_deadzone(joystick.get_axis(0))   # 左搖桿 X
    ly = apply_deadzone(joystick.get_axis(1))   # 左搖桿 Y
    rx = apply_deadzone(joystick.get_axis(3))   # 右搖桿 X
    ry = apply_deadzone(joystick.get_axis(4))   # 右搖桿 Y

    # 位置控制
    vx = -lx * pos_speed
    vy =  ly * pos_speed
    vz = -ry * pos_speed
    cmd_x += vx * dt
    cmd_y += vy * dt
    cmd_z += vz * dt

    # yaw
    vyaw = -rx * rot_speed
    cmd_yaw += vyaw * dt

    # roll：LB / RB
    LB = joystick.get_button(4)
    RB = joystick.get_button(5)
    vroll = 0.0
    if LB:
        vroll = -rot_speed
        cmd_roll += vroll * dt
    if RB:
        vroll = rot_speed
        cmd_roll += vroll * dt

    # pitch：X / Y
    Xbtn = joystick.get_button(2)
    Ybtn = joystick.get_button(3)
    vpitch = 0.0
    if Xbtn:
        vpitch = -rot_speed
        cmd_pitch += vpitch * dt
    if Ybtn:
        vpitch = rot_speed
        cmd_pitch += vpitch * dt

    # 夾爪：A / B
    A = joystick.get_button(0)
    B = joystick.get_button(1)
    vg = 0.0
    if A:
        vg = -grip_speed
        cmd_grip += vg * dt
    if B:
        vg = grip_speed
        cmd_grip += vg * dt

    # 限制邊界（你可以照你 demo 裡的設定調整）
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


########################################
# Policy Eval 在 Real Robot 上
########################################

@click.command()
@click.option('-c', '--checkpoint', required=True, help="Diffusion Policy checkpoint 路徑")
@click.option('-d', '--device', default='cuda:0')
@click.option('-hz', '--control_hz', default=10, type=int)
def main(checkpoint, device, control_hz):
    # -------- 1) 載入 policy --------
    print("📦 Loading checkpoint:", checkpoint)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    print("✅ Policy ready on", device)

    # -------- 2) 啟動 Real Env --------
    env = startEnvNode()
    print("等待 TM Real Env ready...")
    while not env.is_ready():
        time.sleep(0.1)
    print("🤖 TM Real Env Ready!")

    # -------- 3) 搖桿 --------
    joystick = init_joystick()
    if joystick is None:
        return

    rate = control_hz
    dt = 1.0 / rate

    print("==============================================")
    print("🎮 控制方式：")
    print("  Record / BACK  ：模式切換 (手動 ↔ Policy)")
    print("  Menu   / START ：Reset Episode")
    print("==============================================")

    while True:
        # reset episode
        obs = env.reset()
        obs = env.get_obs()

        pos_ee = obs["pos_ee"]
        gripper = float(obs["gripper_length"])

        cmd_x, cmd_y, cmd_z = float(pos_ee[0]), float(pos_ee[1]), float(pos_ee[2])
        cmd_roll, cmd_pitch, cmd_yaw = float(pos_ee[3]), float(pos_ee[4]), float(pos_ee[5])
        cmd_grip = gripper

        pos_speed = 0.1
        rot_speed = 0.5
        grip_speed = 0.06

        policy_running = False
        prev_record_pressed = False
        step_count = 0

        print("\n=== New Episode Started ===")

        while True:
            time.sleep(dt)
            obs = env.get_obs()
            img = obs["img"]
            pos_ee = obs["pos_ee"]
            gripper_length = obs["gripper_length"]
            rgb = obs["rgb"]

            # =========================
            # 讀搖桿 & 模式切換
            # =========================
            # 不管哪個模式，我們都檢查 record / reset
            pygame.event.pump()
            record_btn = joystick.get_button(6)   # BACK
            menu_btn = joystick.get_button(7)     # START

            # reset
            if menu_btn:
                print("🔄 Reset Episode by MENU.")
                break  # 跳出 episode 迴圈 → 重新 reset

            # record rising edge → 模式切換
            record_pressed = bool(record_btn)
            if record_pressed and not prev_record_pressed:
                policy_running = not policy_running
                if policy_running:
                    print("🤖 Policy STARTED.")
                else:
                    print("🖐️ Policy STOPPED, back to joystick control.")
            prev_record_pressed = record_pressed

            # =========================
            # 產生 action
            # =========================
            if policy_running:
                # ---- Policy 模式：忽略搖桿位移，只用模型 ----
                np_obs = {
                    # 根據你的 shape_meta / dataset 調整 key
                    "img": img[None, ...],                      # (1,3,96,96)
                    "pos_ee": pos_ee[None, ...],                # (1,6)
                    "gripper_length": gripper_length[None, ...] # (1,1)
                }

                torch_obs = {
                    k: torch.as_tensor(v, dtype=torch.float32, device=device)
                    for k, v in np_obs.items()
                }

                with torch.no_grad():
                    act_dict = policy.predict_action(torch_obs)

                action = act_dict["action"][0].detach().cpu().numpy()
                velocity = np.zeros_like(action, dtype=np.float32)

            else:
                # ---- 手動模式：跟 demo_tm_real 一樣，用 read_joystick_action ----
                joy = read_joystick_action(
                    joystick, env,
                    cmd_x, cmd_y, cmd_z,
                    cmd_roll, cmd_pitch, cmd_yaw,
                    cmd_grip,
                    dt,
                    pos_speed,
                    rot_speed,
                    grip_speed,
                )

                if joy["need_reset"]:
                    print("🔄 Reset Episode by Joystick (MENU).")
                    break

                action = joy["action"]
                velocity = joy["velocity"]
                (cmd_x, cmd_y, cmd_z,
                 cmd_roll, cmd_pitch, cmd_yaw,
                 cmd_grip) = joy["cmd"]

            # =========================
            # 執行在真實機器人上
            # =========================
            env.exec_action(action, velocity, duration=dt)

            # =========================
            # 視覺化
            # =========================
            vis = rgb.copy()
            vis = cv2.resize(vis, None, fx=1.5, fy=1.5)

            mode_str = "POLICY" if policy_running else "MANUAL"
            mode_color = (0, 255, 0) if policy_running else (0, 0, 255)

            cv2.putText(
                vis,
                f"Mode: {mode_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                mode_color,
                2,
            )
            cv2.putText(
                vis,
                f"Step: {step_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"EE: x={pos_ee[0]:.3f} y={pos_ee[1]:.3f} z={pos_ee[2]:.3f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"Grip: {gripper_length[0]:.3f}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("TM Real Eval Viewer", vis)
            cv2.waitKey(1)

            step_count += 1


if __name__ == "__main__":
    main()
    rclpy.shutdown()
