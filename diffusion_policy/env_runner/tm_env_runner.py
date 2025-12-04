import os
import math
import pathlib
import collections
import dill
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

import numpy as np
import torch

from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class TMPickPlaceRunner(BaseImageRunner):
    """
    跟 PushTImageRunner 類似的 runner，
    只是把環境換成 TMPickPlaceEnv。

    - 會建 n_train + n_test 個 initial condition
    - train/ 開頭是訓練 seed
    - test/  開頭是測試 seed
    - 會自動記錄 reward 與影片到 wandb
    """

    def __init__(
        self,
        output_dir,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        past_action=False,
        n_envs=None,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec

        # env 每幾 step render 一次（給 VideoRecordingWrapper）
        steps_per_render = max(10 // fps, 1)

        # ---------------------------
        # env_fn 建立單一環境
        # ---------------------------
        def env_fn():
            base_env = TMPickPlaceEnv(gui=False)
            venv = VideoRecordingWrapper(
                base_env,
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=crf,
                    thread_type="FRAME",
                    thread_count=1,
                ),
                file_path=None,
                steps_per_render=steps_per_render,
            )
            env = MultiStepWrapper(
                venv,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )
            return env

        env_fns = [env_fn] * n_envs

        # ---------------------------------------------------
        # 準備每個 env 的 seed / prefix / init_fn
        # ---------------------------------------------------
        env_seeds = []
        env_prefixs = []
        env_init_fn_dills = []

        # train 部分
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # env: MultiStepWrapper
                assert isinstance(env, MultiStepWrapper)
                # env.env: VideoRecordingWrapper
                assert isinstance(env.env, VideoRecordingWrapper)

                # reset video path
                env.env.video_recoder.stop()
                env.env.file_path = None

                # 啟用錄影的情況下，設定檔名
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)

                # 設 seed（MultiStepWrapper 會轉呼叫到內部 env 的 seed）
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test 部分
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env, MultiStepWrapper)
                assert isinstance(env.env, VideoRecordingWrapper)

                env.env.video_recoder.stop()
                env.env.file_path = None

                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)

                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # 建立 AsyncVectorEnv
        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills

    # -------------------------------------------------------
    # 跑 evaluation
    # -------------------------------------------------------
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # 讓每個 env 使用對應的 init_fn（設定 seed & 錄影）
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # reset + policy reset
            obs = env.reset()
            policy.reset()
            past_action = None

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval TMPickPlaceRunner {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            step_count = 0
            rewards_this_chunk = []

            while not done and step_count < self.max_steps:
                # 原始 env obs（MultiStepWrapper + AsyncVectorEnv 回傳）
                np_obs_full = dict(obs)

                # 只取出 policy / normalizer 需要的部分
                np_obs_dict = {
                    "img": np_obs_full["img"],
                    "pos_joints": np_obs_full["pos_joints"],
                    "pos_ee": np_obs_full["pos_ee"],
                    "gripper_length": np_obs_full["gripper_length"],
                    "cube_pos": np_obs_full["cube_pos"],
                }

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)


                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                action = np_action_dict["action"]

                obs, reward, done, info = env.step(action)

                # AsyncVectorEnv 回傳的 done / reward shape: (n_envs, n_action_steps)
                # 我們只看 sum reward 就好
                rewards_this_chunk.append(reward.copy())

                done = np.all(done)
                past_action = action
                step_count += action.shape[1]
                pbar.update(action.shape[1])

            pbar.close()

            # 儲存影片路徑 & reward（跟原本 PushTImageRunner 相同介面）
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # 合併這 chunk 內的 reward（沿用 PushT 的 max over time）
            rewards_this_chunk = np.stack(rewards_this_chunk, axis=1)  # (n_envs, T)
            all_rewards[this_global_slice] = rewards_this_chunk[this_local_slice]

        # 清空 buffer
        _ = env.reset()

        # --------------------------
        # logging 到 wandb
        # --------------------------
        max_rewards = collections.defaultdict(list)
        log_data = {}

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            # 取一整個 episode 的最大 reward（跟原始 runner 一樣）
            max_reward = np.max( all_rewards[i] )
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # aggregate
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            log_data[name] = float(np.mean(value))

        return log_data
