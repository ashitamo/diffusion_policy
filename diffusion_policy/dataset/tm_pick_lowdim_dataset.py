from typing import Dict
import copy
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class TMPickLowdimDataset(BaseLowdimDataset):
    """
    TM Pick & Place 的 low-dim dataset
    只使用：
        - state: 你在 demo 裡存的向量 (D_state,)
        - action: 末端目標 (7 維)
    """
    def __init__(self,
                 zarr_path,
                 horizon=16,
                 pad_before=0,
                 pad_after=0,
                 obs_key='state',
                 action_key='action',
                 seed=42,
                 val_ratio=0.1,
                 max_train_episodes=None):
        super().__init__()

        # 只載入 state / action 兩個 key
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[obs_key, action_key]
        )

        # ------- train / val split -------
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # ------- sampler -------
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    # --------------------------------------------------
    # validation dataset
    # --------------------------------------------------
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    # --------------------------------------------------
    # normalizer：對 obs / action 做線性 normalize
    # --------------------------------------------------
    def get_normalizer(self, mode='limits', **kwargs):
        """
        這裡跟 PushTLowdimDataset 一樣，直接對整個 replay_buffer 的
        obs / action 做統計。
        """
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data,
            last_n_dims=1,
            mode=mode,
            **kwargs
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    # --------------------------------------------------
    # 把 sample 轉成 model 需要的格式
    # --------------------------------------------------
    def _sample_to_data(self, sample):
        """
        sample 可能是：
            - sampler 給的 dict（每個 array shape: (T, D)）
            - 或整個 replay_buffer（array shape: (N, T, D)）

        我們只是簡單把 obs = state, action = action
        """
        state = sample[self.obs_key].astype(np.float32)
        action = sample[self.action_key].astype(np.float32)

        data = {
            'obs': state,    # shape: (T, D_state) or (N, T, D_state)
            'action': action # shape: (T, 7) 或 (N, T, 7)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
