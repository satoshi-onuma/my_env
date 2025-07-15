import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt
import os

# Ray RLlibにカスタム環境を正しく認識させるための設定
from ray import tune
# ★ Humanoid環境クラスをインポート
from my_humanoid_env import HumanoidSoftEnv

# ★ Humanoid環境を登録
tune.register_env("HumanoidSoft-v0", lambda config: HumanoidSoftEnv(**config))

# ✅ Ray の初期化
ray.init(logging_level="ERROR")

# ✅ PPO の設定
config = (
    PPOConfig()
    # ★ 学習する環境をHumanoidに変更
    .environment(env="HumanoidSoft-v0")
    .env_runners(
        # Humanoidは複雑なので、並列数を増やすことを推奨
        num_env_runners=8,
        rollout_fragment_length=1000,
    )
    .framework("torch")
    .training(
        # バッチサイズも大きくすると安定しやすい
        train_batch_size=8000,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        entropy_coeff=0.0,
    )
    .resources(
        # GPUがあれば学習が高速化します
        num_gpus=0 # 0のままならCPUを使用
    )
    .debugging(
        seed=42
    )
)

# PPO固有のハイパーパラメータ
config.sgd_minibatch_size = 256
config.num_sgd_iter = 20

# ✅ PPO アルゴリズムインスタンスを構築
print("アルゴリズムを構築中...")
algo = config.build()
print("構築完了。")

# ✅ 学習ループ (★ 複雑なためイテレーション数を増やす)
rewards = []
print("学習を開始します...")
for i in range(100): # 20 -> 100
    result = algo.train()
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
    rewards.append(reward)
    print(f"Iteration {i+1:3d}: Mean Reward = {reward:.2f}")

# ✅ 学習曲線を保存
plt.plot(rewards)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("HumanoidSoft-v0 PPO Training")
plt.grid(True)
plt.savefig("humanoid_soft_training_curve.png")
plt.show()

# ✅ チェックポイント保存 (★ パスをHumanoid用に変更)
checkpoint_dir = os.path.abspath("./humanoid_soft_policy_checkpoint_dir")
checkpoint_result = algo.save(checkpoint_dir)
print(f"✅ ポリシーを保存しました: {checkpoint_result.checkpoint.path}")

# ✅ シャットダウン
ray.shutdown()
print("学習が完了しました。")
