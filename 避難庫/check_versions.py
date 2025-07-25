try:
    import mujoco, gymnasium as gym, torch, ray
    print("MuJoCo:", mujoco.__version__)
    print("Gymnasium:", gym.__version__)
    print("Torch:", torch.__version__, "CUDA Available:", torch.cuda.is_available())
    print("Ray:", ray.__version__)
except Exception as e:
    print("Error occurred:", e)
