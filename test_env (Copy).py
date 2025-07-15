import gymnasium as gym
import my_ant_env  # 登録コードを読み込むために必要

env = AntSoftEnv(render_mode="human")  # または "rgb_array"
obs = env.reset()
for _ in range(100):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
