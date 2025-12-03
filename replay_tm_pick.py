# replay_tm_pick.py
import numpy as np
import time
import cv2
import pybullet as p
import pygame
import click

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv


@click.command()
@click.option('-i', '--input', default="data/tm_pick_demo.zarr", required=True)
@click.option('-hz', '--control_hz', default=30, type=int)
@click.option('-ep', '--episode_id', default=0, type=int)
@click.option('-spd', '--speed_scale', default=1.0, type=float)
def main(input, control_hz, episode_id,speed_scale):
    """
    Replay TM-Pick demo from stored .zarr data.
    用來檢查、驗證 demo 資料是否正確。

    操作：
        Q：離開
        R：重播同一個 episode
        +/-：變更播放速度
    """

    print(f"載入 ReplayBuffer: {input}")
    # ✅ 正確：用 ReplayBuffer API 來讀，而不是自己直接用 zarr
    rb = ReplayBuffer.create_from_path(input, mode='r')
    n_eps = rb.n_episodes
    print(f"共 {n_eps} 個 episodes")

    if n_eps == 0:
        print("❌ 目前沒有任何 episode，可以先用 demo_tm_pick.py 收集資料")
        return

    if episode_id >= n_eps:
        print(f"❌ episode_id={episode_id} 超出範圍，最大是 {n_eps-1}")
        return

    # 先取一次 episode 看 shape
    ep = rb.get_episode(episode_id)
    actions = ep["action"]          # (T, 7)
    states = ep["state"]            # (T, 16) 你之前設定的向量
    imgs = ep["img"]                # (T, 96, 96, 3)
    steps = actions.shape[0]

    print(f"✔ 載入 Episode {episode_id}, 長度 {steps} steps")
    print(f"  action shape: {actions.shape}")
    print(f"  state shape:  {states.shape}")
    print(f"  img shape:    {imgs.shape}")

    # ======================================================
    # 啟動環境
    # ======================================================
    env = TMPickPlaceEnv(rate=control_hz, gui=True)
    dt = 1.0 / control_hz
    
    pygame.init()
    # print(actions[10:18])
    while True:
        print(f"\n=== Replay Episode {episode_id} (speed={speed_scale:.1f}x) ===")
        env.seed(episode_id)
        obs = env.reset()

        for t in range(steps):
            action = actions[t]

            obs, reward, done, info = env.step(action)

            # --- 顯示 OpenCV 大視窗 ---
            vis = obs["rgb"]          # PyBullet 相機畫面

            robot_ee = obs["robot_ee"]
            cube_pos = obs["cube_pos"]

            cv2.putText(vis,
                        f"t={t}/{steps}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            cv2.putText(vis,
                        f"EE: x={robot_ee[0]:.3f} y={robot_ee[1]:.3f} z={robot_ee[2]:.3f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.putText(vis,
                        f"CUBE: x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} z={cube_pos[2]:.3f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 200), 2)

            cv2.imshow("TM Replay Viewer", vis)
            cv2.waitKey(1)

            # ----------------------
            # 控制播放速度
            # ----------------------
            time.sleep(dt / speed_scale)

            # ----------------------
            # Keyboard Events
            # ----------------------
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Exit.")
                        cv2.destroyAllWindows()
                        return
                    if event.key == pygame.K_r:
                        print("Replay from start.")
                        break
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        speed_scale = min(3.0, speed_scale + 0.2)
                        print(f"Speed = {speed_scale:.1f}x")
                    if event.key == pygame.K_MINUS:
                        speed_scale = max(0.2, speed_scale - 0.2)
                        print(f"Speed = {speed_scale:.1f}x")
            else:
                # 沒有 break，就繼續 replay
                continue

            # 有按 R 跳出內層 for → 重播
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
