# diffusion_policy/dataset/tm_pick_image_dataset.py
from typing import Dict
import copy
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class TMPickImageDataset(BaseImageDataset):
    """
    Dataset for TM5 Pick & Place (image + low-dim state).

    讀取你用 demo_tm_pick.py 寫入的 zarr：
        keys: ['img', 'state', 'action', 'keypoint', 'n_contacts']

    這裡我們只用：
        - img: (T, H, W, 3) uint8
        - state: (T, D_state)
        - action: (T, 7)

    輸出給 policy 的格式：
        data = {
            'obs': {
                'image': (T, 3, H, W),
                'state': (T, D_state),
            },
            'action': (T, 7)
        }
    """

    def __init__(self,
                 zarr_path,
                 horizon=16,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.1,
                 max_train_episodes=None):
        super().__init__()

        # 只拿 img/state/action 三個 key 就好
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['img', 'state', 'action']
        )

        # train / val 分割
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

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

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
    # normalizer：對 state / action 做線性 normalize
    # --------------------------------------------------
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],   # (N, T, 7)
            'state': self.replay_buffer['state']      # (N, T, D_state)
        }
        normalizer = LinearNormalizer()
        # last_n_dims=1 表示沿著最後一維做統計
        normalizer.fit(
            data=data,
            last_n_dims=1,
            mode=mode,
            **kwargs
        )
        # 圖像 normalize 用內建的 [-1,1] 範圍
        normalizer['img'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # --------------------------------------------------
    # sampler 回傳一段 sequence，轉成 model 用的格式
    # --------------------------------------------------
    def _sample_to_data(self, sample):
        """
        sample:
            sample['img']   : (T, H, W, 3) uint8
            sample['state'] : (T, D_state)
            sample['action']: (T, 7)
        """
        # image → (T, 3, H, W), [0,1]
        image = sample['img'].astype(np.float32)
        state = sample['state'].astype(np.float32)      # (T, D_state)
        action = sample['action'].astype(np.float32)    # (T, 7)

        data = {
            'obs': {
                'img': image,    # T, 3, H, W
                'state': state,    # T, D_state
            },
            'action': action       # T, 7
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


