import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class AntSoftEnv(MujocoEnv, gym.utils.EzPickle):
    """
    AntSoftEnvクラス
    標準のAnt-v4のロジックをベースに、カスタムXMLを読み込む環境
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        # ★ 修正点: 物理ステップと一致するようにFPSを40に変更
        "render_fps": 40,
    }

    def __init__(self, **kwargs):
        # XMLファイルへのパスを解決
        xml_path = os.path.join(os.path.dirname(__file__), "assets", "ant_soft.xml")
        
        # 観測空間を定義
        # qpos(xy除く): 15-2=13, qvel: 14, contact_forces: 84 => 合計 111
        # ※MuJoCo 3.x以降、cfrc_extはデフォルトで観測に含まれないことが多い
        #   標準のAnt-v5に合わせて、cfrc_extを除いた 27次元の観測空間とします
        #   softblockの関節(1)を追加して28次元
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)

        # 親クラスの初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        gym.utils.EzPickle.__init__(self, **kwargs)
        
        # --- ▼▼▼ デバッグ・検証用の情報を表示 ▼▼▼ ---
        print("--- AntSoftEnv Initialized ---")
        print(f"Model qpos dimension (nq): {self.model.nq}")
        print(f"Model qvel dimension (nv): {self.model.nv}")
        print(f"Observation space dimension: {self.observation_space.shape[0]}")
        
        # soft_slideジョイントがqposのどこに対応するか確認
        try:
            soft_slide_qpos_addr = self.model.joint('soft_slide').qposadr[0]
            print(f"Joint 'soft_slide' is at qpos index: {soft_slide_qpos_addr}")
        except KeyError:
            print("Warning: Joint 'soft_slide' not found.")
        print("-----------------------------")
        # --- ▲▲▲ デバッグ・検証用の情報を表示 ▲▲▲ ---

    def _get_obs(self):
        """
        観測データを取得する。
        Antの位置(x, y)を除いた、体の角度や速度など。
        softblockの関節状態も観測に含める。
        """
        position = self.data.qpos.flat[2:]
        velocity = self.data.qvel.flat
        
        return np.concatenate((position, velocity))

    def _get_reward(self, obs):
        """
        報酬を計算する。
        - 前進報酬: X方向に進んだ距離
        - 生存報酬: 転倒せずに生存している間、常に与えられる
        - コントロールコスト: 大きなトルクを使いすぎることへのペナルティ
        """
        # 前進速度を計算
        x_velocity = (self.data.qpos[0] - self._old_qpos[0]) / self.dt
        
        forward_reward = x_velocity
        # 修正後
        healthy_reward = (not self._check_terminated(obs)) * 1.0
        ctrl_cost = 0.5 * np.square(self.data.ctrl).sum()
        
        reward = forward_reward + healthy_reward - ctrl_cost
        return reward

    def _check_terminated(self, obs):
        """
        エピソードの終了条件をチェックする。
        胴体のZ座標が一定の範囲外になった場合（＝転倒した）に終了
        """
        z_position = self.data.qpos[2]
        is_terminated = not (0.2 <= z_position <= 1.0)
        return is_terminated

    def step(self, action):
        """
        環境を1ステップ進める。
        報酬計算のために、ステップ前の位置情報を保存しておく。
        """
        self._old_qpos = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward(observation)
        terminated = self._check_terminated(observation)
        info = {}

        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, False, info # truncatedは常にFalse

    def reset_model(self):
        """
        エピソード開始時にモデルをリセットする。
        初期姿勢にランダムなノイズを加える。
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
    id="AntSoft-v0",
    entry_point="my_ant_env:AntSoftEnv",
    max_episode_steps=1000,
)
