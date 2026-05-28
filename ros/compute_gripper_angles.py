"""
grasps.csv → パームフレームの yaw/pitch/roll を計算する
Euler角規約: ZYX外因系 (Rz(yaw) @ Ry(pitch) @ Rx(roll))
  yaw  : Z軸周り（ワールド上から見た向き）
  pitch: Y軸周り（前後の傾き）
  roll : X軸周り（左右の傾き）
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

# 23関節インデックス
_J_WRIST      = 0
_J_INDEX_MCP  = 5
_J_MIDDLE_MCP = 8
_J_PINKY_MCP  = 14


def _compute_palm_frame_core(joints: np.ndarray, flip_z: bool):
    wrist      = joints[_J_WRIST]
    index_mcp  = joints[_J_INDEX_MCP]
    middle_mcp = joints[_J_MIDDLE_MCP]
    pinky_mcp  = joints[_J_PINKY_MCP]

    v_index  = index_mcp  - wrist
    v_pinky  = pinky_mcp  - wrist
    v_middle = middle_mcp - wrist

    # Z: cross(v_index, v_pinky) → j5, j14 が XY 平面に乗る
    # 左手: flip_z=False, 右手: flip_z=True (右手では外積が手のひら側を向くため反転)
    z_axis = np.cross(v_index, v_pinky)
    z_axis /= np.linalg.norm(z_axis)
    if flip_z:
        z_axis = -z_axis

    # X: 中指方向を手の平平面に射影 (Gram-Schmidt)
    x_raw  = v_middle / np.linalg.norm(v_middle)
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, wrist, {"x": x_axis, "y": y_axis, "z": z_axis}


def compute_palm_frame_right(joints):
    return _compute_palm_frame_core(joints, flip_z=True)


def compute_palm_frame_left(joints):
    return _compute_palm_frame_core(joints, flip_z=False)


def r_to_euler_zyx_deg(R: np.ndarray):
    """ZYX外因系 Euler角 (yaw, pitch, roll) を degree で返す"""
    rot = Rotation.from_matrix(R)
    yaw, pitch, roll = rot.as_euler('ZYX', degrees=True)
    return yaw, pitch, roll


def main():
    df = pd.read_csv('/home/okada/ws/robo_ros/server/tmp/grasps.csv')
    print(f"rows: {len(df)}\n")

    results = []
    for _, row in df.iterrows():
        joints = np.array(
            [[row[f'j{i}_x'], row[f'j{i}_y'], row[f'j{i}_z']] for i in range(23)],
            dtype=np.float64
        )

        hand = row['hand']
        frame_fn = compute_palm_frame_left if hand == 'left' else compute_palm_frame_right
        R, wrist, axes = frame_fn(joints)

        det = np.linalg.det(R)
        yaw, pitch, roll = r_to_euler_zyx_deg(R)

        print(f"sample={row['sample_idx']}  hand={hand}")
        print(f"  wrist      = [{wrist[0]:.3f}, {wrist[1]:.3f}, {wrist[2]:.3f}]")
        print(f"  palm +X (middle)= [{axes['x'][0]:.3f}, {axes['x'][1]:.3f}, {axes['x'][2]:.3f}]")
        print(f"  palm +Y         = [{axes['y'][0]:.3f}, {axes['y'][1]:.3f}, {axes['y'][2]:.3f}]")
        print(f"  palm +Z (normal)= [{axes['z'][0]:.3f}, {axes['z'][1]:.3f}, {axes['z'][2]:.3f}]")
        print(f"  det(R)     = {det:.6f}  (1.0 が正常)")
        print(f"  yaw  = {yaw:.1f}°  pitch = {pitch:.1f}°  roll = {roll:.1f}°")
        print()

        results.append({
            'sample_idx': row['sample_idx'],
            'hand': hand,
            'yaw_deg': round(yaw, 2),
            'pitch_deg': round(pitch, 2),
            'roll_deg': round(roll, 2),
            'wrist_x': wrist[0], 'wrist_y': wrist[1], 'wrist_z': wrist[2],
            'palm_z_x': axes['z'][0], 'palm_z_y': axes['z'][1], 'palm_z_z': axes['z'][2],
        })

    out = pd.DataFrame(results)
    out_path = '/home/okada/ws/robo_ros/server/tmp/gripper_angles.csv'
    out.to_csv(out_path, index=False)
    print(f"saved → {out_path}")


if __name__ == '__main__':
    main()
