"""
互動版 TM Pick & Place 模擬評估程式

功能：
- 在 PyBullet GUI 中開啟 TMPickPlaceEnv
- 一開始由「人」控制（用 GUI sliders / joystick 或你自己步進）
- 按鍵切換讓「policy」接管控制
- 可以隨時 reset / 結束

鍵盤操作（PyBullet 視窗要在前景）：
  H / h : 切換到 Human control（人控制）
  C / c : 切換到 Policy control（交給 model）
  R / r : reset 環境
  Q / q : 離開程式

Policy control 模式：
- 每次收集最近 n_obs_steps 個 obs
- 用 policy.predict_action(...) 預測一段長度 n_action_steps 的 action 序列
- 依序在 env 裡 step 執行
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import time
import click
import dill
import hydra
import torch
import numpy as np
import pybullet as p
import cv2

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


@click.command()
@click.option("-c", "--checkpoint", required=True,
              help="訓練好的 ckpt 路徑，例如 outputs/.../checkpoints/latest.ckpt")
@click.option("-d", "--device", default="cuda:0",
              help="例如 cuda:0 或 cpu")
@click.option("--rate", default=30, type=int,
              help="PyBullet 模擬頻率 (Hz)")
@click.option("--max_steps", default=500, type=int,
              help="單次 policy control 最多步數（避免卡死）")
def main(checkpoint, device, rate, max_steps):
    # ======================================================
    # 1) 載入 checkpoint & policy（跟 eval.py / eval_real_robot 類似）
    # ======================================================
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 取出 policy (有 EMA 優先用 EMA)
    policy: BaseImagePolicy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        print("[INFO] Using EMA model (EMA weights)")
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # 從 cfg 拿 n_obs_steps / n_action_steps
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    print(f"[INFO] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    # 如果是 diffusion model，可以依照 real robot eval 的習慣調一些推論參數
    if "diffusion" in cfg.name:
        # 視需求調整（也可以保持 ckpt 內原本設定）
        policy.num_inference_steps = getattr(policy, "num_inference_steps", 16)
        policy.n_action_steps = n_action_steps

    # ======================================================
    # 2) 建立 PyBullet GUI 環境
    # ======================================================
    env = TMPickPlaceEnv(rate=rate, gui=True)
    obs = env.reset()

    print("========================================")
    print(" 模擬互動控制說明：")
    print("  H / h : 切換到 Human control（人控制）")
    print("  C / c : 切換到 Policy control（交給 model）")
    print("  R / r : reset 環境")
    print("  Q / q : 離開程式")
    print("========================================")

    dt = 1.0 / rate
    mode = "human"     # 'human' 或 'policy'

    # 用來存最近的 obs 序列（policy 模式要用）
    obs_history = []
    # policy 一次預測的一段 action 序列
    action_buffer = []
    action_step_idx = 0

    # OpenCV 視窗
    cv2.namedWindow("TM Viewer", cv2.WINDOW_NORMAL)

    try:
        while True:
            time.sleep(dt)

            # -----------------------------
            # 讀取鍵盤事件（PyBullet window）
            # -----------------------------
            keys = p.getKeyboardEvents()

            # Quit
            if ord("q") in keys or ord("Q") in keys:
                print("[KEY] Q pressed -> exit.")
                break

            # Reset
            if ord("r") in keys or ord("R") in keys:
                print("[KEY] R pressed -> reset env.")
                obs = env.reset()
                obs_history = []
                action_buffer = []
                action_step_idx = 0
                mode = "human"
                continue

            # 切換到 human mode
            if ord("h") in keys or ord("H") in keys:
                if mode != "human":
                    print("[KEY] Switch to HUMAN control.")
                mode = "human"
                action_buffer = []
                action_step_idx = 0

            # 切換到 policy mode
            if ord("c") in keys or ord("C") in keys:
                if mode != "policy":
                    print("[KEY] Switch to POLICY control.")
                mode = "policy"
                obs_history = []
                action_buffer = []
                action_step_idx = 0
                policy.reset()

            # ==================================================
            # H: Human control 模式
            # ==================================================
            if mode == "human":
                # 在這裡你可以：
                #   - 用 env.read_gui_action() 讀 slider 的 pose + gripper
                #   - 或是用 joystick, 鍵盤等自訂控制方式
                action = env.read_gui_action()
                obs, reward, done, info = env.step(action)

            # ==================================================
            # C: Policy control 模式
            # ==================================================
            elif mode == "policy":
                # 1) 先把當前 obs 加進 history
                obs_history.append(obs)

                # 如果歷史不足 n_obs_steps，就先多跑幾步
                if len(obs_history) < n_obs_steps:
                    # 暫時先讓手臂不動（或用 read_gui_action 也可以）
                    action = env.read_gui_action()
                    obs, reward, done, info = env.step(action)
                else:
                    # 有足夠的 obs 可以餵 model 了
                    # 如果 action_buffer 用完了，就重新規劃一段新的 action 序列
                    if action_step_idx >= len(action_buffer):
                        # 取最近 n_obs_steps 個 obs
                        hist = obs_history[-n_obs_steps:]
                        # print(hist)
                        # ----------------------------
                        # 把 env.obs -> policy 需要的 obs_dict
                        # 這裡要跟你訓練的 dataset / tm_env_runner 一致
                        # ----------------------------
                        # img_seq: (To, 3, H, W)
                        img_seq = np.stack([o["img"] for o in hist], axis=0)

                        # pos_joints: (To, 6)
                        # 如果你的 env.get_obs() 回傳 key 名稱不同（例如 robot_q），
                        # 這裡請改成對應的 key
                        if "pos_joints" in hist[0]:
                            pos_joints_seq = np.stack([o["pos_joints"] for o in hist], axis=0)
                        else:
                            pos_joints_seq = np.stack([o["robot_q"] for o in hist], axis=0)

                        # pos_ee: (To, 6)
                        if "pos_ee" in hist[0]:
                            pos_ee_seq = np.stack([o["pos_ee"] for o in hist], axis=0)
                        else:
                            pos_ee_seq = np.stack([o["robot_ee"] for o in hist], axis=0)

                        # gripper_length: (To, 1)
                        gripper_seq = np.stack([o["gripper_length"] for o in hist], axis=0)

                        cube_pos_seq = np.stack([o["cube_pos"] for o in hist], axis=0)      # (To,6)

                        # 加 batch 維度 -> (B=1, To, ...)
                        img_seq = img_seq[None, ...]                # (1, To, 3, H, W)
                        pos_joints_seq = pos_joints_seq[None, ...]  # (1, To, 6)
                        pos_ee_seq = pos_ee_seq[None, ...]          # (1, To, 6)
                        gripper_seq = gripper_seq[None, ...]        # (1, To, 1)
                        cube_pos_seq = cube_pos_seq[None, ...]      # (1, To, 6)

                        np_obs_dict = {
                            "img": img_seq.astype(np.float32),
                            "pos_joints": pos_joints_seq.astype(np.float32),
                            "pos_ee": pos_ee_seq.astype(np.float32),
                            "gripper_length": gripper_seq.astype(np.float32),
                            "cube_pos": cube_pos_seq.astype(np.float32),
                        }
                        np_obs_dict["img"] = np.moveaxis(np_obs_dict["img"], -1, 2)
                        # numpy -> torch
                        obs_dict = dict_apply(
                            np_obs_dict,
                            lambda x: torch.from_numpy(x).to(device=device),
                        )
                        
                        # ----------------------------
                        # 用 policy 預測一整段 actions
                        # ----------------------------
                        with torch.no_grad():
                            action_dict = policy.predict_action(obs_dict)

                        np_action = (
                            action_dict["action"]
                            .detach()
                            .to("cpu")
                            .numpy()
                        )   # (1, Ta, Da)
                        # 拿出 batch 維度 → (Ta, Da)，轉成 list 好一個一個取
                        action_buffer = list(np_action[0])
                        action_step_idx = 0

                    # 從 buffer 取出這一步要執行的 action
                    action = np.array(action_buffer[action_step_idx], dtype=np.float32)
                    action_step_idx += 1

                    # 執行 env.step
                    obs, reward, done, info = env.step(action)

                    # if done:
                    #     print(f"[INFO] Episode done, reward={reward:.3f}, auto switch to HUMAN.")
                    #     mode = "human"
                    #     obs_history = []
                    #     action_buffer = []
                    #     action_step_idx = 0

                # 簡單保險：避免 policy mode 卡死太久
                max_steps -= 1
                if max_steps <= 0:
                    print("[WARN] Reached max_steps in policy mode, back to HUMAN.")
                    mode = "human"
                    obs_history = []
                    action_buffer = []
                    action_step_idx = 0

            # ==================================================
            # 視覺化：用 env.get_obs() 裡的 rgb 畫到 OpenCV
            # ==================================================
            rgb = obs["rgb"]
            vis = rgb.copy()
            robot_ee = obs["pos_ee"]
            cube_pos = obs["cube_pos"]
            gripper_len = obs["gripper_length"]

            cv2.putText(
                vis,
                f"Mode: {mode.upper()}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"EE: x={robot_ee[0]:.3f} y={robot_ee[1]:.3f} z={robot_ee[2]:.3f}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"CUBE: x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} z={cube_pos[2]:.3f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 200),
                2,
            )
            cv2.putText(
                vis,
                f"GRIPPER: {float(gripper_len[0]):.3f}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 0),
                2,
            )

            cv2.imshow("TM Viewer", vis)
            # 這裡不要用 blocking waitKey，避免卡住 loop
            cv2.waitKey(1)

    finally:
        print("[INFO] Closing...")
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
