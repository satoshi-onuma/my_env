import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import torch
import numpy as np
import imageio
# ユーザーの元のコードにあった、正しいimport文を再度追加
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

# ✅ チェックポイント保存先
checkpoint_dir = os.path.abspath("ant_policy_checkpoint_dir")

# ✅ Ray 初期化
ray.init(logging_level="ERROR") # 警告を減らすためにロギングレベルを設定

# ✅ 学習済みポリシーをロード
print("チェックポイントからポリシーをロード中...")
algo = Algorithm.from_checkpoint(checkpoint_dir)
print("ロード完了。")

# ✅ MuJoCo 環境（録画モード）
env = gym.make("Ant-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)
frames = []

# =================================================================
# ★★★ 推論ループを、あなたの書いた元の正しい方式に戻します ★★★
# =================================================================

# ✅ RLModule取得（新API）
module = algo.get_module("default_policy")

print("推論ループを開始...")
for i in range(1000):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # RLModuleで直接推論
    with torch.no_grad(): # 推論時には勾配計算をオフにする
        fwd_outs = module.forward_inference({"obs": obs_tensor})
    
    # アクションを抽出（あなたの堅牢なロジックをそのまま使用）
    if "actions" in fwd_outs:
        action_tensor = fwd_outs["actions"]
    elif "action_dist_inputs" in fwd_outs:
        action_dist = TorchDiagGaussian.from_logits(fwd_outs["action_dist_inputs"])
        action_tensor = action_dist.sample()
    else:
        keys = list(fwd_outs.keys())
        raise KeyError(f"❌ forward_inference 出力に使えるキーがありません: {keys}")

    action = action_tensor[0].cpu().numpy() # .detach()はno_gradブロック内では不要
    
    # 環境を1ステップ進める
    obs, reward, terminated, truncated, info = env.step(action)

    # 描画してフレームを保存
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    # エピソードが終了したらリセット
    if terminated or truncated:
        print(f"エピソード終了 (ステップ {i+1})。リセットします。")
        obs, info = env.reset()

env.close()
ray.shutdown()

# ✅ フレーム確認と保存
output_path = "ant_policy_demo.mp4"
if not frames:
    raise RuntimeError("❌ 録画フレームが空です。保存できません。")
print(f"✅ {len(frames)} フレームを録画。保存中...")

imageio.mimsave(output_path, frames, fps=30)
print(f"🎥 録画完了: {output_path}")
