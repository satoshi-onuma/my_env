import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import matplotlib.pyplot as plt
import os
import pprint

# ✅ Ray の初期化
ray.init()

# ✅ PPO の設定（Ray 2.46.0 に準拠）
config = (
    PPOConfig()
    .environment(env="Ant-v5")
    .env_runners(
        num_env_runners=4,
        rollout_fragment_length=1000,
    )
    .framework("torch")
    # .training() には、学習のコアなパラメータを渡す
    .training(
        train_batch_size=4000,
        lr=1e-4,
    )
    .resources(
        num_gpus=0
    )
    # seed は .debugging() メソッドで設定する
    .debugging(
        seed=42
    )
)

# PPO固有のハイパーパラメータは、configオブジェクトに直接設定する
config.sgd_minibatch_size = 128
config.num_sgd_iter = 10

# ✅ PPO アルゴリズムインスタンスを構築
algo = config.build()

# ✅ 学習ループ（20イテレーション）
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
plt.title("Ant-v5 PPO Training (Ray 2.46.0)") # バージョンを明記
plt.grid(True)
plt.savefig("ant_training_curve.png")
plt.show()

# ✅ チェックポイント保存
checkpoint_dir = "ant_policy_checkpoint_dir"
checkpoint_result = algo.save(checkpoint_dir)
print(f"✅ ポリシーを保存しました: {checkpoint_result.checkpoint.path}")

# ✅ シャットダウン
ray.shutdown()
print("学習が完了しました。")
