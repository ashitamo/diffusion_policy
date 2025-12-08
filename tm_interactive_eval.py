# tm_interactive_eval.py
"""
讓訓練好的 TM pick diffusion policy 在 PyBullet GUI 裡互動操作。

Usage:
    python tm_interactive_eval.py \
        --checkpoint outputs/2025-12-04/11-38-31/checkpoints/latest.ckpt \
        --device cuda:0
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
import time
import click
import dill
import hydra
import torch
import numpy as np
import pybullet as p

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv


@click.command()
@click.option("-c", "--checkpoint", required=True,
              help="訓練好的 ckpt 路徑")
@click.option("-d", "--device", default="cuda:0",
              help="例如 cuda:0 或 cpu")
@click.option("--rate", default=30, type=int,
              help="PyBullet 模擬頻率 (Hz)")
def main(checkpoint, device, rate):
    # =========================
    # 1) 載入 checkpoint & policy
    # =========================
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]

    # 建立 workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 取出 policy（若有 EMA 就用 EMA）
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        print("[INFO] Using EMA model")
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # 讀 config 裡的 obs/action 步數
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    print(f"[INFO] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    # =========================
    # 2) 建立 GUI 環境
    # =========================
    env = TMPickPlaceEnv(rate=rate, gui=True)
    obs = env.reset()

    print("========================================")
    print(" 互動控制說明：")
    print("   S : 開始由 model 接管控制")
    print("   P/Space : 暫停 / 繼續")
    print("   R : reset 環境")
    print("   Q : 離開程式")
    print("========================================")

    # 用來存最近的 obs 序列
    obs_history = []
    # 用來存一段預先規劃好的 actions（長度 = n_action_steps）
    action_buffer = []
    action_step_idx = 0

    running = False   # 是否由 model 控制中
    dt = 1.0 / rate

    while True:
        time.sleep(dt)

        # =========================
        # 3) Key control (PyBullet)
        # =========================
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
            running = False
            continue

        # Start model control
        if ord("s") in keys or ord("S") in keys:
            print("[KEY] S pressed -> start model control.")
            running = True

        # Pause / resume
        if 32 in keys or ord("p") in keys or ord("P") in keys:
            # 32 = space
            running = not running
            print(f"[KEY] toggle pause, running={running}")
            time.sleep(0.2)  # 簡單防彈跳

        # =========================
        # 4) 若沒有在跑 model，就只維持畫面
        # =========================
        if not running:
            # 這裡可以選擇讓使用者用 GUI slider 控制
            # 或是什麼都不做，單純靜止畫面
            continue

        # =========================
        # 5) 準備 obs 序列，餵給 policy
        # =========================
        # 如果 action_buffer 用完了，就重新規劃一段新的 action 序列
        if action_step_idx >= len(action_buffer):
            # 更新 obs_history
            obs_history.append(obs)
            if len(obs_history) < n_obs_steps:
                # 還不夠長時，用第一個 obs 重複補齊
                pad = [obs_history[0]] * (n_obs_steps - len(obs_history))
                hist = pad + obs_history
            else:
                hist = obs_history[-n_obs_steps:]

            # 組成 image / state 的序列 (To, C, H, W) & (To, D)
            img_seq = np.stack([o["img"] for o in hist], axis=0)      # (To, 3,96,96)
            state_seq = np.stack([o["state"] for o in hist], axis=0)  # (To, D)

            # 加上 batch 維度 -> (B=1, To, ...)
            img_seq = img_seq[None, ...]
            state_seq = state_seq[None, ...]

            np_obs_dict = {
                "img": img_seq.astype(np.float32),
                "state": state_seq.astype(np.float32),
            }

            # numpy -> torch
            obs_dict = {
                k: torch.from_numpy(v).to(device)
                for k, v in np_obs_dict.items()
            }

            # =========================
            # 6) 用 policy 預測一整段 actions
            # =========================
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # 轉回 numpy, 取出 (1, Ta, Da)
            np_action = (
                action_dict["action"]
                .detach()
                .to("cpu")
                .numpy()
            )
            # 轉成 list，之後一個 step 用一個
            action_buffer = list(np_action[0])  # 長度 = n_action_steps
            action_step_idx = 0

        # =========================
        # 7) 取出這一步要執行的 action
        # =========================
        action = np.array(action_buffer[action_step_idx], dtype=np.float32)
        action_step_idx += 1

        # 執行到 env 裡
        obs, reward, done, info = env.step(action)

        # 如果成功 / done，就自動停住，等你按 S 再開始下一回合
        if done:
            print(f"[INFO] Episode done, reward={reward:.3f} -> press R reset, S start again.")
            running = False

    print("[INFO] Closing...")
    env.close()


if __name__ == "__main__":
    main()
