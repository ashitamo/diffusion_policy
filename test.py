from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.tm_pick_image_dataset import TMPickImageDataset
rb = ReplayBuffer.copy_from_path("data/tm_pick_demo.zarr")

img0 = rb['img'][0]
print(img0.shape)          # 應該是 (3, 96, 96)
print(img0.min(), img0.max())

# 簡單測試用
def test():
    import os
    zarr_path = os.path.expanduser('data/tm_pick_demo.zarr')
    dataset = TMPickImageDataset(zarr_path, horizon=16)

    print("n_episodes:", dataset.replay_buffer.n_episodes)
    print("len(dataset):", len(dataset))

    sample = dataset[0]
    print("image shape:", sample['obs']['img'].shape)
    print("cube_pos shape:", sample['obs']['cube_pos'].shape)
    print("pos_joints shape:", sample['obs']['pos_joints'].shape)
    print("pos_ee shape:", sample['obs']['pos_ee'].shape)
    print("gripper_length shape:", sample['obs']['gripper_length'].shape)
    print("action shape:", sample['action'].shape)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.tm_env_pick import TMPickPlaceEnv
import numpy as np

rb = ReplayBuffer.create_from_path("data/tm_pick_demo.zarr", mode='r')
env = TMPickPlaceEnv(rate=30, gui=False)

ep_id = 0
ep = rb.get_episode(ep_id)
demo_state0 = ep['cube_pos'][0]   # (26,)

env.seed(ep_id)
obs = env.reset()
env_state0 = obs['cube_pos']      # (26,)

print("demo_state0:", demo_state0)
print("env_state0 :", env_state0)
print("allclose?  ", np.allclose(demo_state0, env_state0, atol=1e-4))


if __name__ == '__main__':
    test()
