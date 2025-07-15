import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt
import os
import pprint

# Ray RLlibにカスタム環境を正しく認識させるための設定
from ray import tune
from my_ant_env import AntUnevenEnv # 環境クラスを直接インポート

# Rayの登録機能を使ってカスタム環境を登録する
tune.register_env("AntUneven-v0", lambda config: AntUnevenEnv(**config))

# ✅ Ray の初期化
# 多数の警告を非表示にするため、ロギングレベルをERRORに設定
ray.init(logging_level="ERROR")

# ✅ PPO の設定
config = (
    PPOConfig()
    # 学習する環境をカスタム環境に変更
    .environment(env="AntUneven-v0")
    .env_runners(
        num_env_runners=4,
        rollout_fragment_length=1000,
    )
    .framework("torch")
    .training(
        train_batch_size=4000,
        lr=1e-4,
    )
    .resources(
        num_gpus=0
    )
    .debugging(
        seed=42
    )
)

# PPO固有のハイパーパラメータ
config.sgd_minibatch_size = 128
config.num_sgd_iter = 10

# ✅ PPO アルゴリズムインスタンスを構築
print("アルゴリズムを構築中...")
algo = config.build()
print("構築完了。")

# ✅ 学習ループ
rewards = []
print("学習を開始します...")
for i in range(20):
    result = algo.train()
    try:
        reward = result["env_runners"]["episode_return_mean"]
    except KeyError:
        reward = 0.0
    rewards.append(reward)
    print(f"Iteration {i+1:2d}: Mean Reward = {reward:.2f}")

# ✅ 学習曲線を保存
plt.plot(rewards)
plt.xlabel("Iteration")
plt.ylabel("Mean Episode Reward")
plt.title("AntUneven-v0 PPO Training (Ray 2.46.0)")
plt.grid(True)
plt.savefig("ant_uneven_training_curve.png")
plt.show()

# ✅ チェックポイント保存
# ★ 修正: 相対パスから絶対パスに変換して保存エラーを回避
checkpoint_dir = os.path.abspath("./ant_uneven_policy_checkpoint_dir")
checkpoint_result = algo.save(checkpoint_dir)
print(f"✅ ポリシーを保存しました: {checkpoint_result.checkpoint.path}")

# ✅ シャットダウン
ray.shutdown()
print("学習が完了しました。")

