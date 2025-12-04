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
    print("state shape:", sample['obs']['state'].shape)
    print("action shape:", sample['action'].shape)

if __name__ == '__main__':
    test()
