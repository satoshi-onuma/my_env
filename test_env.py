import gymnasium as gym
import my_ant_env  # 登録コードを読み込むために必要

env = gym.make("AntSoft-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
