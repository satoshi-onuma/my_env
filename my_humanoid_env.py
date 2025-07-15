import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class HumanoidSoftEnv(MujocoEnv, gym.utils.EzPickle):
    """
    HumanoidSoftEnvクラス
    新しい複雑なHumanoidモデルを読み込む環境
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, **kwargs):
        # XMLファイルへのパスを解決
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_soft.xml")
        
        # ★★★ 観測空間の次元数を更新 ★★★
        # 新モデルは自由度が高いため、観測空間の次元が増加
        # qpos(root_xy除く): 25-2=23, qvel: 25 => 合計 48次元
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)

        # 親クラスの初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        gym.utils.EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        """
        観測データを取得する。
        """
        position = self.data.qpos.flat[2:]
        velocity = self.data.qvel.flat
        
        return np.concatenate((position, velocity))

    @property
    def contact_forces(self):
        """接触力の大きさを取得"""
        return np.clip(self.data.cfrc_ext, -1, 1).ravel()

    def _get_reward(self, obs):
        """
        報酬を計算する。
        """
        # 前進速度を計算
        x_velocity = (self.data.qpos[0] - self._old_qpos[0]) / self.dt
        
        # 報酬とコスト
        forward_reward = 1.25 * x_velocity
        healthy_reward = 5.0
        ctrl_cost = 0.1 * np.square(self.data.ctrl).sum()
        contact_cost = 0.5 * 1e-3 * np.square(self.contact_forces).sum()
        
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        return reward

    def _check_terminated(self, obs):
        """
        エピソードの終了条件をチェックする。
        """
        z_position = self.data.qpos[2]
        is_terminated = not (1.0 <= z_position <= 2.0)
        return is_terminated

    def step(self, action):
        """
        環境を1ステップ進める。
        """
        self._old_qpos = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward(observation)
        terminated = self._check_terminated(observation)
        info = {}

        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, False, info

    def reset_model(self):
        """
        エピソード開始時にモデルをリセットする。
        """
        noise_low = -0.1
        noise_high = 0.1
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self._old_qpos = self.data.qpos.copy()
        observation = self._get_obs()
        return observation

# グローバルスコープで環境を登録
register(
    id="HumanoidSoft-v0",
    entry_point="my_humanoid_env:HumanoidSoftEnv",
    max_episode_steps=1000,
)
