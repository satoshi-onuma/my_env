import numpy as np

def generate_soft_terrain(
    start_x,          # 地形を配置し始めるX座標
    start_y,          # 地形を配置し始めるY座標
    block_width,      # 個々のブロックのX方向の幅
    block_length,     # 個々のブロックのY方向の長さ
    num_blocks_x,     # X方向に並べるブロックの数
    num_blocks_y,     # Y方向に並べるブロックの数
    block_height=0.01,
    desired_mass=0.001, # ★★★ 密度(density)の代わりに、目標の質量(desired_mass)を引数にする ★★★
    stiffness=5000,
    damping=15
):
    """
    指定されたパラメータに基づいて、MuJoCo用の軟弱地形XMLを生成します。
    """
    worldbody_elements = []
    tendon_elements = []

    # ★★★ ここからが追加・変更部分 ★★★
    # 1. ブロック1個の体積を計算
    volume = block_width * block_length * block_height
    
    # 2. 質量と体積から密度を計算 (密度 = 質量 / 体積)
    #    体積がゼロにならないように、ゼロ除算を避けるためのチェックを追加
    if volume > 1e-9:
        density = desired_mass / volume
    else:
        density = 0
    
    print(f"INFO: Block Volume = {volume:.6f} m^3, Desired Mass = {desired_mass:.6f} kg -> Calculated Density = {density:.3f} kg/m^3")
    # ★★★ ここまでが追加・変更部分 ★★★

    # 3Dモデル用のサイズ（geomのsizeは半分の値を指定するため）
    geom_size_x = block_width / 2.0
    geom_size_y = block_length / 2.0
    geom_size_z = block_height / 2.0

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            # 各ブロックの中心座標を計算
            pos_x = start_x + (i * block_width)  + geom_size_x 
            pos_y = start_y + (j * block_length) + geom_size_y
            pos_z = geom_size_z

            name_suffix = f"{i+1}_{j+1}"

            # ----------------- worldbody に追加する要素 -----------------
            body_xml = f"""
    <body name="softblock_{name_suffix}" pos="{pos_x:.3f} {pos_y:.3f} 0.03">
      <joint name="soft_slide_{name_suffix}" type="slide" axis="0 0 1" damping="2" limited="true" range="-0.15 0.02" armature="0"/>
      <geom name="soft_geom_{name_suffix}" type="box" size="{geom_size_x:.3f} {geom_size_y:.3f} {geom_size_z:.3f}" density="{density:.3f}" rgba="0.9 0.9 0.9 1" material="MatSoftBlock"/>
      <site name="soft_top_{name_suffix}" pos="0 0 {geom_size_z:.3f}" size="0.01" rgba="1 0 0 0.5"/>
    </body>
    <site name="anchor_site_{name_suffix}" pos="{pos_x:.3f} {pos_y:.3f} {0.03+geom_size_z:.3f}" size="0.01" rgba="0 1 0 0.5"/>"""
            worldbody_elements.append(body_xml)
            
            # ----------------- tendon に追加する要素 -----------------
            tendon_xml = f"""
    <spatial name="soft_spring_{name_suffix}" stiffness="{stiffness}" damping="{damping}">
      <site site="soft_top_{name_suffix}"/>
      <site site="anchor_site_{name_suffix}"/>
    </spatial>"""
            tendon_elements.append(tendon_xml)

    # 結果を結合して出力
    print("="*20 + " WORLD BODY ELEMENTS " + "="*20)
    print("")
    for elem in worldbody_elements:
        print(elem)
    print("\n" + "="*23 + " TENDON ELEMENTS " + "="*23)
    print("")
    for elem in tendon_elements:
        print(elem)


if __name__ == '__main__':
    # ==================================================================
    # ここにあるパラメータを自由に変更してください
    # ==================================================================
    
    # 地形を配置し始める左下の座標
    START_X = 0.2
    START_Y = -0.2

    # ブロック1個のサイズ
    BLOCK_WIDTH  = 0.1  # メートル
    BLOCK_LENGTH = 0.1  # メートル

    # X方向とY方向に並べるブロックの数
    NUM_BLOCKS_X = 10
    NUM_BLOCKS_Y = 4
    
    # 床の厚み
    BLOCK_HEIGHT = 0.01 # メートル
    
    # ★★★ 密度(DENSITY)の代わりに、目標の質量(DESIRED_MASS)を指定 ★★★
    DESIRED_MASS = 0.001 # kg
    
    # 床の物理特性
    STIFFNESS = 5000   # バネの硬さ
    DAMPING   = 15    # バネの振動の収まりやすさ
    
    # ==================================================================
    
    generate_soft_terrain(
        start_x=START_X,
        start_y=START_Y,
        block_width=BLOCK_WIDTH,
        block_length=BLOCK_LENGTH,
        num_blocks_x=NUM_BLOCKS_X,
        num_blocks_y=NUM_BLOCKS_Y,
        block_height=BLOCK_HEIGHT,
        desired_mass=DESIRED_MASS,
        stiffness=STIFFNESS,
        damping=DAMPING
    )
