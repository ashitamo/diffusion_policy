from diffusion_policy.common.replay_buffer import ReplayBuffer

rb = ReplayBuffer.copy_from_path("data/tm_pick_demo.zarr", keys=['state'])
print(rb['state'].shape)   # 會是 (N, T, D)