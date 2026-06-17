#!/usr/bin/env python3
"""
sciurus17 把持コントロールパネル (tkinter GUI) — RViz 仮想モード (腰制御付き)

rviz4.py をベースに waist_yaw_joint による腰回転を追加。
アーム移動前に腰を指定角度へ移動することで可動域を拡張し OMPL 計画失敗を低減する。
腰軌道も _plan_steps に保存されるため、main2.py でロードして実機再現可能。

ROS2 + MoveIt2 必須 / 実機ハードウェア不要。
demo.launch.py (use_mock_components:=true) で RViz を起動し、
保存済み RGBD を ROSカメラトピックとして配信してカメラストリームを再現する。

起動 (sciurus17/ ディレクトリで):
  docker compose run --rm sciurus17 \\
      python3 /sciurus17_m2/ros/sciurus17_gui_rviz5.py \\
      --rviz-image /sciurus17_m2/client/saved_data/test_20260422_200416 \\
      --server-url http://10.40.1.115:8082

rviz3 からの変更点:
  - 移動+R6→R7 回転ボタンを追加
    グラスプ位置への移動と R6→R7 リンクベクトルをターゲット把持方向に
    揃える回転を同時に実行する (_exec_move_and_align)。
    Rotation.align_vectors で最小回転を計算し、現在 EEF 姿勢行列に適用。

操作フロー:
  1. [sciurus17 起動] → demo.launch.py (use_mock_components:=true) + RViz 起動
  2. [カメラ接続] → 保存 RGBD を ROS トピックへ配信開始
       RViz の Image Display でも確認可能 (--color-topic のトピック名)
  3. [フレーム取得] → 受信中のフレームを凍結
  4. 凍結映像をクリックして物体選択
  5. [計算機へ送信・姿勢推定] → サーバで pose + 把持姿勢を一括生成
  6. [両アームを移動] → MoveItPy で RViz 仮想ロボットを動かす (軌道が可視化される)

依存:
  ROS2 Jazzy + MoveIt2 + sciurus17_ros (Jazzy ブランチ)
  pip: Pillow opencv-python numpy requests
"""

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow が必要です: pip install Pillow")
    sys.exit(1)

# ── ROS2 (必須) ───────────────────────────────────────────────────────────────
try:
    import rclpy
    import rclpy.duration
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Point, Pose, PointStamped, PoseStamped, Quaternion
    from rclpy.node import Node
    from rclpy.qos import (qos_profile_sensor_data,
                            QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy)
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as RosImage
    from visualization_msgs.msg import Marker, MarkerArray
    from moveit_msgs.msg import DisplayTrajectory
    _ROS2_OK = True
except ImportError as e:
    print(f"[エラー] ROS2 が見つかりません: {e}")
    print("  このスクリプトは ROS2 環境 (Docker コンテナ内) で実行してください。")
    sys.exit(1)

_TF2_OK = False
try:
    import tf2_geometry_msgs  # noqa: F401
    from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
    from geometry_msgs.msg import TransformStamped
    _TF2_OK = True
except ImportError:
    pass

_MOVEIT_OK = False
try:
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy, PlanRequestParameters
    _MOVEIT_OK = True
except ImportError:
    pass

# ── client/ モジュール (省略可能) ─────────────────────────────────────────────
_SERVER_OK = False
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "client"))

try:
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics, ObjectPose, normalized_to_camera
    _SERVER_OK = True
except ImportError as e:
    print(f"[情報] サーバ通信モジュール未読み込み: {e}")

# ── 定数 ──────────────────────────────────────────────────────────────────────
CANVAS_W    = 800
CANVAS_H    = 600
LIVE_MS     = 50
CAM_PUB_HZ  = 10.0   # 仮想カメラの配信レート

S_IDLE       = "idle"
S_CAMERA     = "camera"
S_CAPTURED   = "captured"
S_SELECTED   = "selected"
S_ESTIMATING = "estimating"
S_MOVING     = "moving"
S_DONE       = "done"

BG       = "#2b2b2b"
PANEL_BG = "#3c3f41"
BTN_BG   = "#555759"
BTN_FG   = "white"
LOG_BG   = "#1e1e1e"
LOG_FG   = "#cccccc"


# ── ユーティリティ ─────────────────────────────────────────────────────────────

def _euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> "Quaternion":
    """RPY (ZYX convention) → geometry_msgs/Quaternion"""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )



def _gravity_from_neck_pitch(neck_pitch_deg: float) -> np.ndarray:
    theta = math.radians(neck_pitch_deg)
    return np.array([0.0, -math.cos(theta), -math.sin(theta)], dtype=np.float32)


# ── パームフレーム → グリッパ回転角 ────────────────────────────────────────────
# カメラ光学座標系 (x=右, y=下, z=奥) → base_link (x=前, y=左, z=上) 回転行列
_R_CAM_OPT_TO_BASE = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

# 初期位置: 肘曲げ・+X 前方・XY 平面平行 (base_link [m])
# 肩 z≈0.29 に揃えて水平にし、x 方向に前方伸展 → MoveIt が肘曲げIK解を選ぶ
_HOME_L = np.array([0.35,  0.15, 0.30])
_HOME_R = np.array([0.35, -0.15, 0.30])

# 初期姿勢 EEF 回転行列 (base_link)
# 爪(gripper_y)が +X を向く水平姿勢
# 左: EEF_x→+Z, EEF_y→+X(爪), EEF_z→+Y  q=(x=-0.5,y=-0.5,z=-0.5,w=0.5)
# 右: EEF_x→-Y, EEF_y→+X(爪), EEF_z→+Z  q=(x=0,y=0,z=-0.707,w=0.707)  Rz(-90°)
_R_HOME_L = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]], dtype=np.float64)
_R_HOME_R = np.array([[ 0, 1, 0],
                       [-1, 0, 0],
                       [ 0, 0, 1]], dtype=np.float64)

_J_G_WRIST      = 0
_J_G_INDEX_MCP  = 5
_J_G_MIDDLE_MCP = 8
_J_G_PINKY_MCP  = 14


def _compute_palm_frame(joints: np.ndarray, flip_z: bool) -> np.ndarray:
    """23関節座標からパームフレーム回転行列 R (3x3, カメラ座標系) を返す。
    左手: flip_z=False  右手: flip_z=True
    """
    wrist  = joints[_J_G_WRIST]
    v_idx  = joints[_J_G_INDEX_MCP]  - wrist
    v_pin  = joints[_J_G_PINKY_MCP]  - wrist
    v_mid  = joints[_J_G_MIDDLE_MCP] - wrist
    z_axis = np.cross(v_idx, v_pin)
    z_axis /= np.linalg.norm(z_axis)
    if flip_z:
        z_axis = -z_axis
    x_raw  = v_mid / np.linalg.norm(v_mid)
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _palm_to_gripper_euler(joints: np.ndarray, flip_z: bool, r_home: np.ndarray):
    """カメラ座標系の23関節 → 初期姿勢(中指→+X)基準の相対 ZYX Euler角 [deg]
    r_home: 初期姿勢の EEF 回転行列 (_R_HOME_L / _R_HOME_R)
    yaw=pitch=roll=0 ↔ 中指ベクトルが +X を向く初期姿勢と一致
    """
    R_cam  = _compute_palm_frame(joints, flip_z)
    R_base = _R_CAM_OPT_TO_BASE @ R_cam
    R_rel  = r_home.T @ R_base
    yaw, pitch, roll = Rotation.from_matrix(R_rel).as_euler('ZYX', degrees=True)
    return float(yaw), float(pitch), float(roll)


def _project(xyz, intr):
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z <= 1e-6:
        return None
    return (int(intr.fx * x / z + intr.cx), int(intr.fy * y / z + intr.cy))


def _draw_pose_axes(rgb_bgr: np.ndarray, R: np.ndarray,
                    t: np.ndarray, intr, axis_len: float = 0.08):
    out = rgb_bgr.copy()
    if intr is None:
        return out
    origin = _project(t, intr)
    if origin is None:
        return out
    for i, (label, color) in enumerate([
        ("X", (0,   0,   255)),
        ("Y", (0,   255, 0  )),
        ("Z", (255, 0,   0  )),
    ]):
        tip = _project(np.asarray(t) + np.asarray(R)[:, i] * axis_len, intr)
        if tip:
            cv2.arrowedLine(out, origin, tip, color, 2, tipLength=0.3)
            cv2.putText(out, label, (tip[0] + 4, tip[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.circle(out, origin, 6, (255, 255, 255), -1)
    return out


_HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 18),
    (0, 5),  (5, 6),  (6, 7),  (7, 19),
    (0, 8),  (8, 9),  (9, 10), (10, 20),
    (0, 11), (11,12), (12,13), (13,21),
    (0, 14), (14,15), (15,16), (16,17), (17,22),
]


def _draw_hand_skeleton(rgb_bgr: np.ndarray, lh_cam: np.ndarray,
                         rh_cam: np.ndarray, intr) -> np.ndarray:
    out = rgb_bgr.copy()
    if intr is None:
        return out
    lh_cam = np.asarray(lh_cam, dtype=np.float64)
    rh_cam = np.asarray(rh_cam, dtype=np.float64)
    for joints, color_bone, color_joint, label in [
        (lh_cam, (0,   0, 200), (0,   0, 255), "L"),
        (rh_cam, (180, 0,   0), (255, 0,   0), "R"),
    ]:
        if joints.ndim != 2:
            continue
        pts2d = [_project(joints[i], intr) for i in range(len(joints))]
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts2d) and j < len(pts2d) and pts2d[i] and pts2d[j]:
                cv2.line(out, pts2d[i], pts2d[j], color_bone, 2, cv2.LINE_AA)
        for k, p in enumerate(pts2d):
            if p:
                cv2.circle(out, p, 6 if k == 0 else 3, color_joint, -1, cv2.LINE_AA)
        if pts2d[0]:
            cv2.putText(out, label, (pts2d[0][0] + 8, pts2d[0][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_joint, 2)
    return out


# ── 可視化ウィンドウ ───────────────────────────────────────────────────────────

class VisualizationWindow:
    PW, PH = 480, 360

    _PANELS = [
        ("camera", "カメラ映像",   0, 0),
        ("frozen", "物体選択",     0, 1),
        ("pose",   "姿勢推定結果", 1, 0),
        ("grasp",  "把持姿勢結果", 1, 1),
    ]

    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("sciurus17 Estimation Viewer")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)

        self._canvases: dict[str, tk.Canvas] = {}
        self._tk_imgs:  dict[str, object]    = {}

        for key, label, row, col in self._PANELS:
            frame = tk.Frame(self.win, bg=BG)
            frame.grid(row=row, column=col, padx=4, pady=(4, 0))
            tk.Label(frame, text=label, bg=BG, fg="#bbbbbb",
                     font=("Helvetica", 9, "bold")).pack(anchor="w")
            c = tk.Canvas(frame, width=self.PW, height=self.PH, bg="#111111")
            c.pack()
            c.create_text(self.PW // 2, self.PH // 2, text="待機中",
                          fill="#444444", font=("Helvetica", 12))
            self._canvases[key] = c
            self._tk_imgs[key]  = None

    def _show(self, key: str, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(self.PW / w, self.PH / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (nw, nh))
        pad = np.zeros((self.PH, self.PW, 3), dtype=np.uint8)
        y0, x0 = (self.PH - nh) // 2, (self.PW - nw) // 2
        pad[y0:y0+nh, x0:x0+nw] = rgb
        tk_img = ImageTk.PhotoImage(Image.fromarray(pad))
        self._tk_imgs[key] = tk_img
        c = self._canvases[key]
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=tk_img)

    def update_camera(self, bgr: np.ndarray):
        self._show("camera", bgr)

    def update_frozen(self, bgr: np.ndarray, click_x: int = -1, click_y: int = -1):
        img = bgr.copy()
        if click_x >= 0:
            cv2.circle(img, (click_x, click_y), 14, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (click_x, click_y),  4, (80, 80, 255), -1)
        cv2.putText(img, "FROZEN", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 80), 2)
        self._show("frozen", img)

    def update_pose(self, bgr: np.ndarray, R: np.ndarray, t: np.ndarray, intr):
        img = _draw_pose_axes(bgr, R, t, intr)
        cv2.putText(img, f"t [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]",
                    (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        self._show("pose", img)

    def update_grasp(self, bgr: np.ndarray, lh_cam: np.ndarray,
                     rh_cam: np.ndarray, intr):
        lh_cam = np.asarray(lh_cam, dtype=np.float64)
        rh_cam = np.asarray(rh_cam, dtype=np.float64)
        img = _draw_hand_skeleton(bgr, lh_cam, rh_cam, intr)
        lh_w = lh_cam[0] if lh_cam.ndim == 2 else lh_cam
        rh_w = rh_cam[0] if rh_cam.ndim == 2 else rh_cam
        for i, (v, lbl) in enumerate([(lh_w, "L"), (rh_w, "R")]):
            cv2.putText(img, f"{lbl}[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]",
                        (6, img.shape[0] - 8 - i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        self._show("grasp", img)


class LogWindow:
    """ログ専用ウィンドウ。閉じても withdraw するだけで再表示できる。"""

    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("sciurus17 Log")
        self.win.configure(bg=BG)
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)

        btn_frame = tk.Frame(self.win, bg=BG)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 0))
        tk.Button(btn_frame, text="クリア", command=self._clear,
                  bg=BTN_BG, fg=BTN_FG, relief=tk.FLAT,
                  font=("Helvetica", 9), padx=6).pack(side=tk.RIGHT)

        self._area = scrolledtext.ScrolledText(
            self.win, width=100, height=30,
            bg=LOG_BG, fg=LOG_FG, font=("Courier", 9),
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self._area.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def append(self, msg: str):
        self._area.configure(state=tk.NORMAL)
        self._area.insert(tk.END, msg)
        self._area.see(tk.END)
        self._area.configure(state=tk.DISABLED)

    def _clear(self):
        self._area.configure(state=tk.NORMAL)
        self._area.delete("1.0", tk.END)
        self._area.configure(state=tk.DISABLED)

    def show(self):
        self.win.deiconify()
        self.win.lift()


def _remove_display_motion_path(cfg: dict):
    """MoveItPy の config dict から DisplayMotionPath アダプターを再帰的に除去する。"""
    for key, value in cfg.items():
        if isinstance(value, dict):
            _remove_display_motion_path(value)
        elif isinstance(value, list):
            cfg[key] = [v for v in value if 'DisplayMotionPath' not in str(v)]
        elif isinstance(value, str) and 'DisplayMotionPath' in value:
            cfg[key] = ' '.join(v for v in value.split() if 'DisplayMotionPath' not in v)


def _plan_and_execute(robot, comp, log_fn, params, traj_out=None) -> bool:
    result = comp.plan(single_plan_parameters=params)
    if result:
        robot.execute(result.trajectory, controllers=[])
        if traj_out is not None:
            traj_out.append(result.trajectory)
        return True
    log_fn("軌道計画失敗")
    return False


# ── RViz 仮想カメラノード ──────────────────────────────────────────────────────

class RvizRobotNode(Node):
    """
    RViz 仮想モード用 ROS2 ノード。

    保存済み RGBD を ROS カメラトピックとして配信し、同ノードでサブスクライブする。
    これにより RViz の Image Display からも同じ映像を確認できる。
    TF2 変換は実際の ROS TF ツリーを使用し、失敗時は固定オフセットで補完する。
    MoveItPy による軌道計画・実行は RViz で可視化される。
    """

    def __init__(self, image_dir: str | None,
                 color_topic: str, depth_topic: str, info_topic: str,
                 camera_frame: str):
        super().__init__("sciurus17_rviz_gui_node")
        self._bridge        = CvBridge()
        self._lock          = threading.Lock()
        self._camera_frame  = camera_frame

        # 保存済み RGBD (配信元データ)
        self._saved_rgb:      np.ndarray | None = None
        self._saved_depth_mm: np.ndarray | None = None
        self._saved_cam_json: dict | None       = None

        # サブスクライブで受け取ったデータ (GUI が参照する)
        self._latest_rgb:    np.ndarray | None = None
        self._latest_depth:  np.ndarray | None = None
        self._intrinsics                       = None

        self._publishing = False

        self._tf_fallback_logged = False
        if _TF2_OK:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._static_tf_br = StaticTransformBroadcaster(self)
            # spin 開始後に Static TF を送信するためタイマー経由にする
            self._static_tf_timer = self.create_timer(0.5, self._send_static_tf_once)

        # ── パブリッシャ (sensor_data = BEST_EFFORT: RViz と QoS 互換) ──
        self._pub_color = self.create_publisher(RosImage,   color_topic, qos_profile_sensor_data)
        self._pub_depth = self.create_publisher(RosImage,   depth_topic, qos_profile_sensor_data)
        self._pub_info  = self.create_publisher(CameraInfo, info_topic,  qos_profile_sensor_data)

        # ── サブスクライバ (自身が配信したものを受け取る) ──
        self.create_subscription(RosImage,   color_topic, self._color_cb, qos_profile_sensor_data)
        self.create_subscription(RosImage,   depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  qos_profile_sensor_data)

        # 配信タイマ (start_camera() 後に有効)
        self._timer = self.create_timer(1.0 / CAM_PUB_HZ, self._publish_camera)

        # 把持目標マーカ (TRANSIENT_LOCAL = ラッチ: 後からサブスクライブしても受信できる)
        _marker_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._pub_markers = self.create_publisher(MarkerArray, "/grasp_markers", _marker_qos)
        self._pub_joint_markers = self.create_publisher(MarkerArray, "/joint_markers", _marker_qos)
        # 起動後 1s で DELETEALL を送り、RViz がトピックを認識できるようにする
        self._marker_init_timer = self.create_timer(1.0, self._init_markers_once)

        # RViz のゴール姿勢ゴースト消去用 (TRANSIENT_LOCAL で確実に受信させる)
        self._pub_display_path = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', _marker_qos)
        # 起動後 1s で前回セッションのオレンジゴーストを消去する
        self._display_path_init_timer = self.create_timer(1.0, self._init_display_path_once)

        self._load_saved_image(image_dir)

    # ──────────────────────── Static TF (仮想カメラフレーム) ─────────────────────

    def _send_static_tf_once(self):
        """spin 開始後 0.5s で一度だけ呼ばれ、カメラ Static TF を登録する。"""
        self._publish_camera_static_tf()
        self._static_tf_timer.cancel()

    def _init_markers_once(self):
        """起動後 1s で DELETEALL を送り /grasp_markers トピックを RViz に認識させる。"""
        arr = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        arr.markers.append(m)
        self._pub_markers.publish(arr)
        self._marker_init_timer.cancel()
        self.get_logger().info("[marker] 起動初期化: /grasp_markers DELETEALL 送信済み")

    def _init_display_path_once(self):
        """起動後 1s で前回セッションのオレンジゴースト (planned path) を消去する。"""
        self.clear_planned_path()
        self._display_path_init_timer.cancel()
        self.get_logger().info("[display_path] 起動初期化: /display_planned_path クリア済み")

    def _publish_camera_static_tf(self):
        """head_camera_color_optical_frame を TF ツリーへ登録する (base_link 基準固定値)。"""
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id  = self._camera_frame
        t.transform.translation.x = 0.30
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.20
        t.transform.rotation.w = 1.0
        self._static_tf_br.sendTransform(t)
        self.get_logger().info(
            f"[TF] Static TF 送信: 'base_link' → '{self._camera_frame}' (+0.30, 0, +0.20)"
        )

    # ──────────────────────── 保存画像ロード ───────────────────────────────────

    def _load_saved_image(self, image_dir: str | None):
        dirs_to_try = []
        if image_dir:
            dirs_to_try.append(image_dir)
        for rel in (
            "client/saved_data/test_20260422_200416",
            "client/saved_data/test_20260422_200438",
        ):
            dirs_to_try.append(os.path.join(_REPO_ROOT, rel))

        for d in dirs_to_try:
            p = os.path.join(d, "rgb.png")
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    self._saved_rgb = img
                    self.get_logger().info(f"[RViz] 画像ロード: {p}")
                    dp = os.path.join(d, "depth.png")
                    if os.path.exists(dp):
                        self._saved_depth_mm = cv2.imread(dp, cv2.IMREAD_ANYDEPTH)
                    cp = os.path.join(d, "cam.json")
                    if os.path.exists(cp):
                        with open(cp) as f:
                            self._saved_cam_json = json.load(f)
                    return

        self.get_logger().warn("[RViz] 保存画像なし。フォールバック画像を配信します。")
        self._saved_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self._saved_rgb, "RViz Mode — No Saved Image",
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    # ──────────────────────── カメラ配信 ───────────────────────────────────────

    def start_camera(self):
        """カメラ接続ボタン押下時に呼ぶ。配信を開始する。"""
        self._publishing = True

    def _publish_camera(self):
        """タイマコールバック。保存 RGBD を ROS トピックへ配信する。"""
        if not self._publishing:
            return
        now = self.get_clock().now().to_msg()

        if self._saved_rgb is not None:
            msg = self._bridge.cv2_to_imgmsg(self._saved_rgb, "bgr8")
            msg.header.stamp    = now
            msg.header.frame_id = self._camera_frame
            self._pub_color.publish(msg)

        if self._saved_depth_mm is not None:
            msg = self._bridge.cv2_to_imgmsg(self._saved_depth_mm, "16UC1")
            msg.header.stamp    = now
            msg.header.frame_id = self._camera_frame
            self._pub_depth.publish(msg)

        self._pub_info.publish(self._make_camera_info_msg(now))

    def _make_camera_info_msg(self, stamp) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._camera_frame

        if self._saved_cam_json:
            K = self._saved_cam_json["cam_K"]
            msg.width  = self._saved_cam_json.get("width",  640)
            msg.height = self._saved_cam_json.get("height", 480)
            fx, fy     = float(K[0]), float(K[4])
            cx, cy     = float(K[2]), float(K[5])
        else:
            h, w       = self._saved_rgb.shape[:2] if self._saved_rgb is not None else (480, 640)
            msg.width, msg.height = w, h
            fx = fy = 600.0
            cx, cy  = w / 2.0, h / 2.0

        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx,  0.0, cx,  0.0, fy,  cy,  0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx,  0.0, cx,  0.0, 0.0, fy,  cy,  0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    # ──────────────────────── サブスクライバコールバック ───────────────────────

    def _color_cb(self, msg: RosImage):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._latest_rgb = bgr
        except Exception as e:
            self.get_logger().error(f"color_cb: {e}")

    def _depth_cb(self, msg: RosImage):
        try:
            d = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self._latest_depth = d.copy()
        except Exception as e:
            self.get_logger().error(f"depth_cb: {e}")

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            if self._intrinsics is None and _SERVER_OK:
                self._intrinsics = CameraIntrinsics(
                    fx=msg.k[0], fy=msg.k[4],
                    cx=msg.k[2], cy=msg.k[5],
                    width=msg.width, height=msg.height,
                )

    # ──────────────────────── GUI インタフェース ────────────────────────────────

    def get_latest_rgb(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_rgb.copy() if self._latest_rgb is not None else None

    def get_snapshot(self) -> tuple:
        with self._lock:
            rgb   = self._latest_rgb.copy()   if self._latest_rgb   is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            intr  = self._intrinsics
        if rgb is None:
            h, w = 480, 640
            rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        if depth is None:
            h, w = rgb.shape[:2]
            depth = np.zeros((h, w), dtype=np.uint16)
        return rgb, depth, intr

    def camera_to_base(self, xyz_cam, camera_frame: str, base_frame: str,
                       x_offset: float = -0.5,
                       y_offset: float = 0.0,
                       z_offset: float = 0.10) -> np.ndarray:
        """カメラ光学座標系 → base_link 近似変換 (RViz 仮想モード固定)。
        camera optical: x=右, y=下, z=奥行き
        base_link:      x=前, y=左, z=上
        x/y/z_offset は base_link 座標系での加算オフセット。
        """
        cx, cy, cz = float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])
        xyz = np.array([
            cz + x_offset,   # 奥行き → 前方 + X オフセット
            -cx + y_offset,  # 右方   → 左方 + Y オフセット
            -cy + z_offset,  # 下方   → 上方 + Z オフセット
        ])
        self.get_logger().info(
            f"[coord] cam({cx:+.3f},{cy:+.3f},{cz:+.3f}) "
            f"offset=({x_offset:+.3f},{y_offset:+.3f},{z_offset:+.3f}) "
            f"→ base({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})"
        )
        return xyz

    def publish_grasp_markers(self, lh_base: np.ndarray, rh_base: np.ndarray):
        """左右手首目標位置を RViz マーカ (球体 + テキスト) として配信する。"""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        configs = [
            (0, lh_base, (1.0, 0.3, 0.1), "L"),   # 赤: 左手首
            (1, rh_base, (0.1, 0.5, 1.0), "R"),   # 青: 右手首
        ]
        for idx, xyz, (r, g, b), label in configs:
            # 球体マーカ
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = now
            m.ns              = "grasp_sphere"
            m.id              = idx
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = float(xyz[0])
            m.pose.position.y = float(xyz[1])
            m.pose.position.z = float(xyz[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.85
            m.lifetime.sec = 0
            arr.markers.append(m)

            # テキストラベル (球の少し上)
            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp    = now
            t.ns              = "grasp_label"
            t.id              = idx
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x = float(xyz[0])
            t.pose.position.y = float(xyz[1])
            t.pose.position.z = float(xyz[2]) + 0.07
            t.pose.orientation.w = 1.0
            t.scale.z         = 0.04
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            t.text            = f"{label}[{xyz[0]:+.2f},{xyz[1]:+.2f},{xyz[2]:+.2f}]"
            t.lifetime.sec    = 0
            arr.markers.append(t)

        self._pub_markers.publish(arr)
        self.get_logger().info("[marker] 把持目標マーカを /grasp_markers へ配信しました")

    def publish_hand_keypoints_markers(
        self,
        lh_cam_all,
        rh_cam_all,
        camera_frame: str,
        base_frame: str,
        x_offset: float = -0.5,
        y_offset: float = 0.0,
        z_offset: float = 0.10,
    ):
        """手首(j0)と指先5本(j18-j22)を base_link フレームでマーカ配信する。"""
        # joint_idx, label, R, G, B  ※手首は publish_grasp_markers で配信済みのため除外
        KEYPOINTS = [
            (18, "thumb",  (1.0, 0.5, 0.0)),
            (19, "index",  (0.2, 0.4, 1.0)),
            (20, "middle", (0.9, 0.1, 0.1)),
            (21, "ring",   (0.1, 0.7, 0.1)),
            (22, "pinky",  (0.6, 0.1, 0.8)),
        ]

        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        mid = 100  # publish_grasp_markers の 0,1 と競合しないよう 100 から

        for hand_prefix, cam_all in [("L", lh_cam_all), ("R", rh_cam_all)]:
            if cam_all is None or np.asarray(cam_all).ndim != 2 or np.asarray(cam_all).shape[0] < 23:
                mid += len(KEYPOINTS) * 2
                continue
            cam_all = np.asarray(cam_all, dtype=np.float64)
            for joint_idx, kp_label, color in KEYPOINTS:
                r, g, b = color
                xyz_base = self.camera_to_base(
                    cam_all[joint_idx], camera_frame, base_frame,
                    x_offset, y_offset, z_offset,
                )

                m = Marker()
                m.header.frame_id = "base_link"
                m.header.stamp    = now
                m.ns              = "hand_kp_sphere"
                m.id              = mid
                m.type            = Marker.SPHERE
                m.action          = Marker.ADD
                m.pose.position.x = float(xyz_base[0])
                m.pose.position.y = float(xyz_base[1])
                m.pose.position.z = float(xyz_base[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.025
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.9
                m.lifetime.sec = 0
                arr.markers.append(m)

                t = Marker()
                t.header.frame_id = "base_link"
                t.header.stamp    = now
                t.ns              = "hand_kp_label"
                t.id              = mid
                t.type            = Marker.TEXT_VIEW_FACING
                t.action          = Marker.ADD
                t.pose.position.x = float(xyz_base[0])
                t.pose.position.y = float(xyz_base[1])
                t.pose.position.z = float(xyz_base[2]) + 0.04
                t.pose.orientation.w = 1.0
                t.scale.z         = 0.025
                t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
                t.text            = f"{hand_prefix}_{kp_label}"
                t.lifetime.sec    = 0
                arr.markers.append(t)
                mid += 1

        self._pub_markers.publish(arr)
        self.get_logger().info("[marker] 手首・指先キーポイントマーカを /grasp_markers へ配信しました")

    def publish_home_direction_markers(self,
                                       lh_pos: np.ndarray, rh_pos: np.ndarray,
                                       lh_xaxis: "np.ndarray | None" = None,
                                       rh_xaxis: "np.ndarray | None" = None,
                                       arrow_len: float = 0.15):
        """手首位置から中指ベクトル(x軸)方向の矢印マーカを配信する。
        xaxis 省略時は初期方向 (1,0,0) を使用。"""
        _X = np.array([1.0, 0.0, 0.0])
        lh_xaxis = np.asarray(lh_xaxis, dtype=float) if lh_xaxis is not None else _X
        rh_xaxis = np.asarray(rh_xaxis, dtype=float) if rh_xaxis is not None else _X
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for mid, wrist, xaxis, color in [
            (200, lh_pos, lh_xaxis, (1.0, 1.0, 0.0)),   # 左: 黄
            (201, rh_pos, rh_xaxis, (0.0, 1.0, 1.0)),   # 右: シアン
        ]:
            tip = wrist + xaxis * arrow_len
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = now
            m.ns              = "home_direction"
            m.id              = mid
            m.type            = Marker.ARROW
            m.action          = Marker.ADD
            m.points = [
                Point(x=float(wrist[0]), y=float(wrist[1]), z=float(wrist[2])),
                Point(x=float(tip[0]),   y=float(tip[1]),   z=float(tip[2])),
            ]
            m.scale.x = 0.012
            m.scale.y = 0.024
            m.scale.z = 0.0
            r, g, b = color
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            m.lifetime.sec = 0
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp    = now
            t.ns              = "home_direction_label"
            t.id              = mid
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x = float(tip[0]) + 0.02
            t.pose.position.y = float(tip[1])
            t.pose.position.z = float(tip[2]) + 0.03
            t.pose.orientation.w = 1.0
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            t.text = "+X"
            t.lifetime.sec = 0
            arr.markers.append(t)
        self._pub_markers.publish(arr)
        self.get_logger().info("[marker] 中指方向マーカ (+X) を /grasp_markers へ配信しました")

    def publish_goal_direction_markers(self,
                                      lh_pos: np.ndarray, rh_pos: np.ndarray,
                                      lh_xaxis: "np.ndarray | None" = None,
                                      rh_xaxis: "np.ndarray | None" = None,
                                      arrow_len: float = 0.15):
        """ゴール把持姿勢の手首位置からゴール中指ベクトル(x軸)方向の矢印マーカを配信する。"""
        _X = np.array([1.0, 0.0, 0.0])
        lh_xaxis = np.asarray(lh_xaxis, dtype=float) if lh_xaxis is not None else _X
        rh_xaxis = np.asarray(rh_xaxis, dtype=float) if rh_xaxis is not None else _X
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for mid, wrist, xaxis, color, label in [
            (202, lh_pos, lh_xaxis, (1.0, 0.5, 0.0), "GL"),   # 左ゴール: オレンジ
            (203, rh_pos, rh_xaxis, (1.0, 0.0, 1.0), "GR"),   # 右ゴール: マゼンタ
        ]:
            tip = wrist + xaxis * arrow_len
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = now
            m.ns              = "goal_direction"
            m.id              = mid
            m.type            = Marker.ARROW
            m.action          = Marker.ADD
            m.points = [
                Point(x=float(wrist[0]), y=float(wrist[1]), z=float(wrist[2])),
                Point(x=float(tip[0]),   y=float(tip[1]),   z=float(tip[2])),
            ]
            m.scale.x = 0.012
            m.scale.y = 0.024
            m.scale.z = 0.0
            r, g, b = color
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            m.lifetime.sec = 0
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp    = now
            t.ns              = "goal_direction_label"
            t.id              = mid
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x = float(tip[0]) + 0.02
            t.pose.position.y = float(tip[1])
            t.pose.position.z = float(tip[2]) + 0.03
            t.pose.orientation.w = 1.0
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = 1.0; t.color.a = 1.0
            t.text = label
            t.lifetime.sec = 0
            arr.markers.append(t)
        self._pub_markers.publish(arr)
        self.get_logger().info("[marker] ゴール中指方向マーカを /grasp_markers へ配信しました")

    def read_eef_pose(self, robot, pose_link: str):
        """FK で EEF の位置 (3,) と回転行列 (3,3) を返す。"""
        with robot.get_planning_scene_monitor().read_only() as scene:
            rs = scene.current_state
            rs.update()
            tf_mat = np.asarray(rs.get_global_link_transform(pose_link))
        pos   = tf_mat[:3, 3]
        r_mat = tf_mat[:3, :3]
        return pos, r_mat

    def read_link_direction(self, robot, link6_name: str, pose_link: str):
        """FK で EEF 位置 (3,) と j6→j7 単位ベクトル (3,) を返す。"""
        with robot.get_planning_scene_monitor().read_only() as scene:
            rs = scene.current_state
            rs.update()
            tf6 = np.asarray(rs.get_global_link_transform(link6_name))
            tf7 = np.asarray(rs.get_global_link_transform(pose_link))
        pos  = tf7[:3, 3]
        vec  = tf7[:3, 3] - tf6[:3, 3]
        norm = np.linalg.norm(vec)
        vec  = vec / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        return pos, vec

    def publish_joint_markers(self, robot):
        """FK で各関節リンク位置を /joint_markers へ球体+ラベルで配信する。"""
        try:
            with robot.get_planning_scene_monitor().read_only() as scene:
                rs = scene.current_state
                rs.update()

                arr = MarkerArray()
                now = self.get_clock().now().to_msg()
                mid = 0

                for prefix, r_c, g_c, b_c in [
                    ("l", 0.1, 0.9, 0.1),   # 左: 緑
                    ("r", 0.9, 0.5, 0.1),   # 右: 橙
                ]:
                    for i in range(1, 8):
                        link_name = f"{prefix}_link{i}"
                        try:
                            tf_mat = rs.get_global_link_transform(link_name)
                        except Exception:
                            mid += 1
                            continue
                        x = float(tf_mat[0, 3])
                        y = float(tf_mat[1, 3])
                        z = float(tf_mat[2, 3])

                        # 球体 (EEF=joint7 は少し大きく)
                        m = Marker()
                        m.header.frame_id = "base_link"
                        m.header.stamp = now
                        m.ns = "joint_sphere"
                        m.id = mid
                        m.type = Marker.SPHERE
                        m.action = Marker.ADD
                        m.pose.position.x = x
                        m.pose.position.y = y
                        m.pose.position.z = z
                        m.pose.orientation.w = 1.0
                        m.scale.x = m.scale.y = m.scale.z = 0.04 if i == 7 else 0.025
                        m.color.r = r_c; m.color.g = g_c; m.color.b = b_c
                        m.color.a = 0.85
                        m.lifetime.sec = 0
                        arr.markers.append(m)

                        # ラベル
                        t = Marker()
                        t.header.frame_id = "base_link"
                        t.header.stamp = now
                        t.ns = "joint_label"
                        t.id = mid
                        t.type = Marker.TEXT_VIEW_FACING
                        t.action = Marker.ADD
                        t.pose.position.x = x
                        t.pose.position.y = y
                        t.pose.position.z = z + 0.04
                        t.pose.orientation.w = 1.0
                        t.scale.z = 0.025
                        t.color.r = t.color.g = t.color.b = 1.0
                        t.color.a = 1.0
                        t.text = f"{prefix.upper()}{i}({x:+.2f},{y:+.2f},{z:+.2f})"
                        t.lifetime.sec = 0
                        arr.markers.append(t)

                        mid += 1

            self._pub_joint_markers.publish(arr)
            self.get_logger().info("[marker] 関節マーカを /joint_markers へ配信しました")
        except Exception as e:
            self.get_logger().warn(f"[marker] 関節マーカ配信失敗: {e}")

    def clear_planned_path(self):
        """RViz のゴール姿勢ゴースト（計画済み軌道表示）を消去する。"""
        self._pub_display_path.publish(DisplayTrajectory())

    def move_waist(self, robot, angle_rad: float, traj_out=None) -> bool:
        """waist_yaw_joint を指定角度 (rad) に移動する。"""
        from moveit.core.robot_state import RobotState
        waist = robot.get_planning_component("waist_group")
        model = robot.get_robot_model()
        rs = RobotState(model)
        rs.set_joint_group_positions("waist_group", [angle_rad])
        waist.set_start_state_to_current_state()
        waist.set_goal_state(robot_state=rs)
        from moveit.planning import PlanRequestParameters
        p = PlanRequestParameters(robot, "ompl_rrtc_default")
        p.max_velocity_scaling_factor     = 0.1
        p.max_acceleration_scaling_factor = 0.1
        p.planning_time                   = 5.0
        ok = _plan_and_execute(robot, waist, self.get_logger().error, p, traj_out=traj_out)
        time.sleep(0.2)
        self.clear_planned_path()
        return ok

    def move_arm(self, robot, arm_comp, params, xyz_base, pose_link: str,
                 orientation: "Quaternion | None" = None, traj_out=None) -> bool:
        """MoveItPy で目標位置へ移動。orientation 省略時は現在の向きを FK で維持。"""
        if orientation is None:
            with robot.get_planning_scene_monitor().read_only() as scene:
                rs = scene.current_state
                rs.update()
                tf_mat = rs.get_global_link_transform(pose_link)
            rot = Rotation.from_matrix(np.asarray(tf_mat)[:3, :3].astype(np.float64))
            qx, qy, qz, qw = rot.as_quat()
        else:
            qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        goal = PoseStamped()
        goal.header.frame_id = "base_link"
        goal.pose = Pose(
            position=Point(x=float(xyz_base[0]),
                           y=float(xyz_base[1]),
                           z=float(xyz_base[2])),
            orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
        )
        arm_comp.set_start_state_to_current_state()
        arm_comp.set_goal_state(pose_stamped_msg=goal, pose_link=pose_link)
        ok = _plan_and_execute(robot, arm_comp, self.get_logger().error, params, traj_out=traj_out)
        time.sleep(0.3)
        self.clear_planned_path()
        return ok

    def move_arm_wrist(self, robot, arm_comp, params,
                       roll: float, pitch: float, yaw: float,
                       pose_link: str) -> bool:
        """EEF の向きを base_link 座標系の絶対 RPY で指定。位置は現在値を維持。
        Yaw → Yaw+Pitch → Yaw+Pitch+Roll の順に Cartesian IK で逐次実行。
        roll/pitch/yaw はラジアン単位。
        """
        with robot.get_planning_scene_monitor().read_only() as scene:
            rs = scene.current_state
            rs.update()
            tf_mat = rs.get_global_link_transform(pose_link)
        x, y, z = float(tf_mat[0, 3]), float(tf_mat[1, 3]), float(tf_mat[2, 3])
        self.get_logger().info(
            f"[wrist] {pose_link} 現在位置: ({x:.3f}, {y:.3f}, {z:.3f})")

        steps = []
        eps = math.radians(0.01)

        def add_step(r, p, yw, label):
            if steps:
                prev = steps[-1]
                if (abs(prev[0] - r) < eps and
                        abs(prev[1] - p) < eps and
                        abs(prev[2] - yw) < eps):
                    return
            steps.append((r, p, yw, label))

        if abs(yaw) >= eps:
            add_step(0.0, 0.0, yaw, f"Yaw={math.degrees(yaw):.1f}°")
        if abs(pitch) >= eps:
            add_step(0.0, pitch, yaw, f"+Pitch={math.degrees(pitch):.1f}°")
        add_step(
            roll, pitch, yaw,
            f"target RPY={math.degrees(roll):.1f},"
            f"{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}°",
        )

        for step_idx, (r, p, yw, label) in enumerate(steps):
            q = _euler_to_quaternion(math.degrees(r), math.degrees(p), math.degrees(yw))
            goal = PoseStamped()
            goal.header.frame_id = "base_link"
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = z
            goal.pose.orientation = q

            self.get_logger().info(f"[wrist] step{step_idx + 1}/{len(steps)} {label}")
            arm_comp.set_start_state_to_current_state()
            arm_comp.set_goal_state(pose_stamped_msg=goal, pose_link=pose_link)
            ok = _plan_and_execute(robot, arm_comp, self.get_logger().error, params)
            time.sleep(0.3)
            if not ok:
                self.get_logger().error(f"[wrist] {label} 失敗 — 中断")
                return False

        self.get_logger().info("[wrist] target reached; holding current arm pose")
        self.clear_planned_path()
        return True


# ── GUI ───────────────────────────────────────────────────────────────────────

class SciurusRvizGUI:
    """sciurus17 把持コントロールパネル — RViz 仮想モード (tkinter)"""

    def __init__(self, root: tk.Tk, node: RvizRobotNode, args):
        self.root = root
        self.node = node
        self.args = args

        self._state   = S_IDLE
        self._working = False

        self._captured_rgb:   np.ndarray | None = None
        self._captured_depth: np.ndarray | None = None
        self._intrinsics                        = None
        self._click_x = self._click_y = -1
        self._scale_x = self._scale_y = 1.0

        self._mesh_path:    str | None       = None
        self._sam6d_client                   = None
        self._R:    np.ndarray | None        = None
        self._t:    np.ndarray | None        = None
        self._gravity: np.ndarray | None     = None
        self._lh_cam:  np.ndarray | None     = None
        self._rh_cam:  np.ndarray | None     = None
        self._lh_base: np.ndarray | None     = None
        self._rh_base: np.ndarray | None     = None
        self._lh_cam_all: np.ndarray | None  = None
        self._rh_cam_all: np.ndarray | None  = None
        self._lh_goal_xaxis: np.ndarray | None = None
        self._rh_goal_xaxis: np.ndarray | None = None
        self._plan_steps: list = []
        self._waist_angle_rad: float = 0.0
        self._robot                          = None
        self._process: subprocess.Popen | None = None
        self._vis_win: VisualizationWindow | None = None

        self._log_q: queue.Queue        = queue.Queue()
        self._btns: dict[str, tk.Button] = {}
        self._log_win: LogWindow | None  = None

        self._build_ui()
        self._on_neck_pitch_change()
        self._open_log_win()
        self._log("[モード] RViz 仮想モード — ROS2 + MoveIt2 (mock_components)")
        self._log(f"  カラートピック: {args.color_topic}")
        self._log(f"  深度トピック:   {args.depth_topic}")
        self._log("  [sciurus17 起動] ボタンで demo.launch.py を起動してください。")
        self._refresh_ui()
        self._open_vis_win()
        self._poll_live()
        self._poll_log()

    def _open_log_win(self):
        if self._log_win is None:
            self._log_win = LogWindow(self.root)
        else:
            self._log_win.show()

    def _open_vis_win(self):
        self._vis_win = VisualizationWindow(self.root)

    def _on_neck_pitch_change(self, *_):
        pitch = self._neck_pitch_var.get()
        self._gravity = _gravity_from_neck_pitch(pitch)
        g = self._gravity
        if self._lbl_gravity is not None:
            self._lbl_gravity.config(text=f"[{g[0]:+.3f}  {g[1]:+.3f}  {g[2]:+.3f}]")

    # ──────────────────────── UI 構築 ──────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.title("sciurus17 Grasp Control Panel [RViz]")
        root.configure(bg=BG)
        root.resizable(True, True)

        # ── ヘルパ ───────────────────────────────────────────────────────────────
        def add_btn(parent, label, cmd, key, *, color=BTN_BG):
            b = tk.Button(parent, text=label, command=cmd,
                          bg=color, fg=BTN_FG,
                          activebackground="#6e6e6e", activeforeground="white",
                          relief=tk.FLAT, font=("Helvetica", 10),
                          padx=4, pady=4, anchor="w")
            b.pack(fill=tk.X, pady=2)
            self._btns[key] = b
            return b

        def section(title):
            f = tk.LabelFrame(bottom, text=title, bg=PANEL_BG, fg="#bbbbbb",
                              font=("Helvetica", 9, "bold"),
                              padx=6, pady=4, relief=tk.GROOVE)
            f.pack(side=tk.LEFT, fill=tk.Y, padx=3, pady=3, anchor="n")
            return f

        def spinrow(parent, label, var, from_, to_, incr=5.0, fmt=None, cmd=None):
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg=PANEL_BG, fg="#aaaaaa",
                     font=("Helvetica", 8)).pack(anchor="w")
            kw = {"command": cmd} if cmd else {}
            if fmt:
                kw["format"] = fmt
            tk.Spinbox(row, textvariable=var, from_=from_, to=to_, increment=incr,
                       width=7, bg=BTN_BG, fg=BTN_FG, font=("Helvetica", 9),
                       **kw).pack(anchor="w", padx=2)

        # ── タイトル ─────────────────────────────────────────────────────────────
        tk.Label(root,
                 text="sciurus17 把持コントロールパネル  【RViz 仮想モード】",
                 font=("Helvetica", 13, "bold"), bg=BG, fg="#44ccff",
                 ).pack(fill=tk.X, pady=(8, 2), padx=10)

        # ── カメラ映像 ───────────────────────────────────────────────────────────
        cam_lf = tk.LabelFrame(root,
                                text="ROS カメラトピック映像  ←  凍結後にクリックで物体選択",
                                bg=BG, fg="#bbbbbb", font=("Helvetica", 9, "bold"))
        cam_lf.pack(padx=8)

        self._canvas = tk.Canvas(cam_lf, width=CANVAS_W, height=CANVAS_H, bg="black")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas_img_id = None

        # ── 下部パネル (セクション横並び) ────────────────────────────────────────
        bottom = tk.Frame(root, bg=BG)
        bottom.pack(fill=tk.X, padx=8, pady=(4, 8))

        # 1. 起動
        f1 = section("1. 起動")
        add_btn(f1, "▶  sciurus17 起動", self._on_launch, "launch", color="#3d6b3d")
        add_btn(f1, "■  停止",           self._on_kill,   "kill",   color="#6b3d3d")
        tk.Frame(f1, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        add_btn(f1, "○  グリッパ 開",      self._on_gripper_open,  "g_open")
        add_btn(f1, "●  グリッパ 閉",      self._on_gripper_close, "g_close")
        tk.Frame(f1, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        add_btn(f1, "⌂  初期姿勢へ (両)",  self._on_home,       "home")
        add_btn(f1, "⌂  左初期姿勢",       self._on_home_left,  "home_left")
        add_btn(f1, "⌂  右初期姿勢",       self._on_home_right, "home_right")

        # 2. カメラ・頭部制御
        f2 = section("2. カメラ・頭部制御")
        add_btn(f2, "▶  カメラ接続", self._on_connect_camera, "cam_connect")
        tk.Frame(f2, height=1, bg="#555555").pack(fill=tk.X, pady=4)
        self._neck_yaw_var   = tk.DoubleVar(value=0.0)
        self._neck_pitch_var = tk.DoubleVar(value=getattr(self.args, "neck_pitch", 0.0))
        spinrow(f2, "ヨー [deg] (左+):",   self._neck_yaw_var,   -90, 90)
        spinrow(f2, "ピッチ [deg] (下+):", self._neck_pitch_var, -90, 90,
                cmd=self._on_neck_pitch_change)
        add_btn(f2, "↓  首を動かす",    self._on_head_move, "head_move")
        add_btn(f2, "↑  首を初期位置へ", self._on_head_home, "head_home")
        grav_row = tk.Frame(f2, bg=PANEL_BG)
        grav_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(grav_row, text="重力:", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._lbl_gravity = tk.Label(grav_row, text="計算中...",
                                      bg=PANEL_BG, fg="#88aaff", font=("Courier", 8))
        self._lbl_gravity.pack(side=tk.LEFT, padx=4)
        tk.Frame(f2, height=1, bg="#555555").pack(fill=tk.X, pady=4)
        add_btn(f2, "フレーム取得",  self._on_capture,   "capture")
        add_btn(f2, "Log",          self._open_log_win, "log_btn")

        # 3. 姿勢推定・座標補正
        f4 = section("3. 姿勢推定・補正")
        tk.Label(f4, text="画像クリックで物体を選択",
                 bg=PANEL_BG, fg="#888888", font=("Helvetica", 8)).pack(anchor="w")
        add_btn(f4, "↩  別の物体を選択", self._on_reselect, "reselect")
        tk.Frame(f4, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        add_btn(f4, "→  計算機へ送信・姿勢推定", self._on_estimate, "estimate")
        tk.Frame(f4, height=1, bg="#555555").pack(fill=tk.X, pady=6)
        self._x_offset_var = tk.DoubleVar(value=-0.5)
        self._y_offset_var = tk.DoubleVar(value=0.0)
        self._z_offset_var = tk.DoubleVar(value=0.10)
        spinrow(f4, "X オフセット [m] (前方+):", self._x_offset_var, -3.0, 3.0, 0.05, "%.2f")
        spinrow(f4, "Y オフセット [m] (左+):",   self._y_offset_var, -3.0, 3.0, 0.05, "%.2f")
        spinrow(f4, "Z オフセット [m] (上+):",   self._z_offset_var, -2.0, 2.0, 0.05, "%.2f")
        add_btn(f4, "↻  再計算・マーカ更新", self._on_recalc, "recalc", color="#4a5a3a")

        # 4. 腰制御
        f_waist = section("4. 腰制御 (waist_yaw_joint)")
        self._waist_angle_var = tk.DoubleVar(value=0.0)
        self._waist_auto_var  = tk.BooleanVar(value=True)
        spinrow(f_waist, "腰 Yaw [deg] (左+/右-):", self._waist_angle_var, -90.0, 90.0, 5.0, "%.1f")
        chk_row = tk.Frame(f_waist, bg=PANEL_BG)
        chk_row.pack(anchor="w")
        tk.Checkbutton(
            chk_row, text="アーム移動前に腰を自動移動",
            variable=self._waist_auto_var,
            bg=PANEL_BG, fg=BTN_FG, selectcolor="#555555",
            activebackground=PANEL_BG, activeforeground=BTN_FG,
        ).pack(side=tk.LEFT)
        btn_row = tk.Frame(f_waist, bg=PANEL_BG)
        btn_row.pack(anchor="w", fill=tk.X)
        add_btn(btn_row, "↻  腰を移動",   self._on_waist_move,  "waist_move",  color="#5a4a2a")
        add_btn(btn_row, "⌂  腰初期化",   self._on_waist_home,  "waist_home",  color="#4a4a2a")
        for b in (self._btns.get("waist_move"), self._btns.get("waist_home")):
            if b:
                b.pack_configure(side=tk.LEFT, fill=tk.X, expand=True)

        # 5. アーム制御（3列: 共通ボタン / 左アーム方向 / 右アーム方向）
        f5 = section("5. アーム制御")
        f5_inner = tk.Frame(f5, bg=PANEL_BG)
        f5_inner.pack(fill=tk.BOTH)

        f5_col0 = tk.Frame(f5_inner, bg=PANEL_BG)
        f5_col0.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        f5_col1 = tk.Frame(f5_inner, bg=PANEL_BG)
        f5_col1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        f5_col2 = tk.Frame(f5_inner, bg=PANEL_BG)
        f5_col2.pack(side=tk.LEFT, fill=tk.Y)

        # 列0: 共通ボタン
        add_btn(f5_col0, "▶  両アームを移動",      self._on_move,              "move",              color="#3d5b7a")
        add_btn(f5_col0, "▶  左アームのみ",      self._on_move_left,         "move_left",         color="#3a547a")
        add_btn(f5_col0, "▶  右アームのみ",      self._on_move_right,        "move_right",        color="#3a547a")
        tk.Frame(f5_col0, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        add_btn(f5_col0, "⇒  両: 移動+R6→R7回転", self._on_move_align_both,  "move_align_both",   color="#2d5a6a")
        add_btn(f5_col0, "⇒  左: 移動+R6→R7回転", self._on_move_align_left,  "move_align_left",   color="#2a4a6a")
        add_btn(f5_col0, "⇒  右: 移動+R6→R7回転", self._on_move_align_right, "move_align_right",  color="#2a4a6a")
        tk.Frame(f5_col0, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        add_btn(f5_col0, "⊕  関節マーカ更新",  self._on_joint_markers, "joint_markers", color="#5a4a2a")
        tk.Frame(f5_col0, height=1, bg="#555555").pack(fill=tk.X, pady=3)
        save_row = tk.Frame(f5_col0, bg=PANEL_BG)
        save_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(save_row, text="保存先:", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._save_path_var = tk.StringVar(value="/sciurus17_m2/grasp_plan.plan")
        tk.Entry(save_row, textvariable=self._save_path_var,
                 bg="#444444", fg="white", font=("Courier", 8),
                 width=22).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        add_btn(f5_col0, "↓  計画を保存",  self._on_save_plan,  "save_plan",  color="#5a4a1a")
        add_btn(f5_col0, "↺  計画クリア",  self._on_clear_plan, "clear_plan", color="#5a3a1a")

        # 列1: 左アーム
        self._l_yaw_var   = tk.DoubleVar(value=0.0)
        self._l_pitch_var = tk.DoubleVar(value=0.0)
        self._l_roll_var  = tk.DoubleVar(value=0.0)
        tk.Label(f5_col1, text="左アーム (j5/j6/j7 直接指定)", bg=PANEL_BG, fg="#aaffaa",
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        spinrow(f5_col1, "j5 Roll  [deg]:", self._l_roll_var,  -180, 180, 5.0, "%.0f")
        spinrow(f5_col1, "j6 Pitch [deg]:", self._l_pitch_var, -180, 180, 5.0, "%.0f")
        spinrow(f5_col1, "j7 Yaw   [deg]:", self._l_yaw_var,   -180, 180, 5.0, "%.0f")
        add_btn(f5_col1, "↻  左アーム回転", self._on_rotate_left, "rotate_left", color="#3a5a4a")

        # 列2: 右アーム
        self._r_yaw_var   = tk.DoubleVar(value=0.0)
        self._r_pitch_var = tk.DoubleVar(value=0.0)
        self._r_roll_var  = tk.DoubleVar(value=0.0)
        tk.Label(f5_col2, text="右アーム (j5/j6/j7 直接指定)", bg=PANEL_BG, fg="#ffaaaa",
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        spinrow(f5_col2, "j5 Roll  [deg]:", self._r_roll_var,  -180, 180, 5.0, "%.0f")
        spinrow(f5_col2, "j6 Pitch [deg]:", self._r_pitch_var, -180, 180, 5.0, "%.0f")
        spinrow(f5_col2, "j7 Yaw   [deg]:", self._r_yaw_var,   -180, 180, 5.0, "%.0f")
        add_btn(f5_col2, "↻  右アーム回転", self._on_rotate_right, "rotate_right", color="#5a3a4a")

        # 把持座標 + ステータス
        fr = section("把持座標 [m]")
        self._lbl_lh    = tk.Label(fr, text="左手首: -", bg=PANEL_BG, fg="#aaffaa",
                                    font=("Courier", 9), anchor="w")
        self._lbl_lh.pack(fill=tk.X)
        self._lbl_rh    = tk.Label(fr, text="右手首: -", bg=PANEL_BG, fg="#aaffaa",
                                    font=("Courier", 9), anchor="w")
        self._lbl_rh.pack(fill=tk.X)
        self._lbl_click = tk.Label(fr, text="選択点: -", bg=PANEL_BG, fg="#ffaaaa",
                                    font=("Courier", 9), anchor="w")
        self._lbl_click.pack(fill=tk.X)
        self._lbl_status = tk.Label(fr, text="● 待機中", bg=PANEL_BG, fg="#888888",
                                     font=("Helvetica", 9), anchor="w")
        self._lbl_status.pack(fill=tk.X, pady=(8, 0))

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────── ログ ─────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_q.put(f"[{ts}] {msg}\n")

    def _poll_log(self):
        try:
            while True:
                msg = self._log_q.get_nowait()
                if self._log_win is not None:
                    self._log_win.append(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    # ──────────────────────── ライブ映像 ───────────────────────────────────────

    def _poll_live(self):
        if self._state == S_CAMERA:
            frame = self.node.get_latest_rgb()
            if frame is not None:
                self._show_frame(frame, overlay_text="ROS TOPIC")
                if self._vis_win:
                    self._vis_win.update_camera(frame)
        self.root.after(LIVE_MS, self._poll_live)

    def _show_frame(self, bgr: np.ndarray,
                    markers: list | None = None,
                    overlay_text: str | None = None):
        h, w = bgr.shape[:2]
        self._scale_x = CANVAS_W / w
        self._scale_y = CANVAS_H / h
        disp = cv2.resize(bgr, (CANVAS_W, CANVAS_H))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        if overlay_text:
            cv2.putText(disp, overlay_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 220, 80), 2, cv2.LINE_AA)
        if markers:
            for ix, iy, color in markers:
                cx, cy = int(ix * self._scale_x), int(iy * self._scale_y)
                cv2.circle(disp, (cx, cy), 14, color, 2, cv2.LINE_AA)
                cv2.circle(disp, (cx, cy),  4, color, -1)
                cv2.putText(disp, f"({ix},{iy})", (cx + 16, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        pil = Image.fromarray(disp)
        self._tk_img = ImageTk.PhotoImage(pil)
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            self._canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

    # ──────────────────────── キャンバスクリック ────────────────────────────────

    def _on_canvas_click(self, event):
        if self._state not in (S_CAPTURED, S_SELECTED, S_DONE):
            return
        img_x = max(0, int(event.x / self._scale_x))
        img_y = max(0, int(event.y / self._scale_y))
        self._click_x, self._click_y = img_x, img_y
        self._lbl_click.config(text=f"選択点: ({img_x}, {img_y})")
        self._log(f"物体選択: ({img_x}, {img_y})")
        if self._captured_rgb is not None:
            self._show_frame(self._captured_rgb,
                             markers=[(img_x, img_y, (255, 80, 80))],
                             overlay_text="FROZEN")
            if self._vis_win:
                self._vis_win.update_frozen(self._captured_rgb, img_x, img_y)
        self._set_state(S_SELECTED)

    # ──────────────────────── ステート管理 ─────────────────────────────────────

    def _set_state(self, s: str):
        self._state = s
        self.root.after(0, self._refresh_ui)

    def _refresh_ui(self):
        s, w = self._state, self._working
        can_arm = not w and s not in (S_IDLE,)

        enabled = {
            "launch":      True,
            "kill":        True,
            "cam_connect": s == S_IDLE,
            "capture":     s == S_CAMERA and not w,
            "head_move":   can_arm,
            "head_home":   can_arm,
            "reselect":    s in (S_CAPTURED, S_SELECTED, S_DONE) and not w,
            "estimate":    s == S_SELECTED and not w,
            "recalc":      s == S_DONE and not w,
            "move":              s == S_DONE and not w,
            "log_btn":           True,
            "move_left":         s == S_DONE and not w,
            "move_right":        s == S_DONE and not w,
            "move_align_both":   s == S_DONE and not w,
            "move_align_left":   s == S_DONE and not w,
            "move_align_right":  s == S_DONE and not w,
            "rotate_left":       s == S_DONE and not w,
            "rotate_right":      s == S_DONE and not w,
            "g_open":          can_arm,
            "g_close":         can_arm,
            "home":            can_arm,
            "home_left":       can_arm,
            "home_right":      can_arm,
            "joint_markers":   can_arm,
            "waist_move":      can_arm,
            "waist_home":      can_arm,
            "save_plan":       s == S_DONE and not w,
            "clear_plan":      not w,
        }
        for key, btn in self._btns.items():
            btn.config(state=tk.NORMAL if enabled.get(key, False) else tk.DISABLED)

        STATUS = {
            S_IDLE:       ("● 待機中",                                  "#888888"),
            S_CAMERA:     ("● ROS トピック配信中",                       "#88cc88"),
            S_CAPTURED:   ("● フレーム取得済 ─ 画像をクリックして選択",   "#88aaff"),
            S_SELECTED:   ("● 物体選択済 ─ 計算機へ送信してください",      "#ffcc44"),
            S_ESTIMATING: ("⏳ 推定・把持姿勢生成中...",                   "#ffaa44"),
            S_MOVING:     ("⏳ アーム移動中 (RViz 確認)...",              "#ff8844"),
            S_DONE:       ("✓ 完了 ─ アームを移動できます",               "#88ff88"),
        }
        text, color = STATUS.get(s, ("●", "white"))
        self._lbl_status.config(text=text, fg=color)
        self._canvas.config(cursor="crosshair" if s in (S_CAPTURED, S_SELECTED, S_DONE) else "")

    # ──────────────────────── バックグラウンド実行 ──────────────────────────────

    def _run_bg(self, func, *a, on_error_state: str | None = None):
        self._working = True
        self._refresh_ui()

        def wrapper():
            try:
                func(*a)
            except Exception as e:
                self._log(f"[エラー] {e}")
                if on_error_state is not None:
                    self.root.after(0, lambda: self._set_state(on_error_state))
            finally:
                self._working = False
                self.root.after(0, self._refresh_ui)

        threading.Thread(target=wrapper, daemon=True).start()

    # ──────────────────────── ボタンハンドラ ───────────────────────────────────

    _MOVEIT_READY_KEYWORDS = (
        "All requested controllers are active",
        "You can start planning now!",
        "MoveGroup action server ready",
        "Ready to take commands for planning group",
    )

    def _on_launch(self):
        cmd = self.args.launch_cmd
        if not cmd:
            self._log("[起動] --launch-cmd が未指定です。")
            return
        self._log(f"[起動] {cmd}")
        try:
            self._process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            def _read():
                home_triggered = False
                for line in self._process.stdout:
                    self._log(f"[ROS] {line.rstrip()}")
                    if not home_triggered and any(kw in line for kw in self._MOVEIT_READY_KEYWORDS):
                        home_triggered = True
                        self._log("[起動] MoveIt2 準備完了 — 必要なら [⌂ 初期姿勢へ] を押してください")
            threading.Thread(target=_read, daemon=True).start()
        except Exception as e:
            self._log(f"[起動エラー] {e}")

    def _on_kill(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log("[停止] sciurus17 プロセスを終了しました")
        else:
            self._log("[停止] 起動済みプロセスがありません")

    def _on_connect_camera(self):
        self._log("仮想カメラトピックの配信を開始します...")
        self._log(f"  RViz の [Image Display] でも確認できます: {self.args.color_topic}")
        self.node.start_camera()
        self._set_state(S_CAMERA)

    def _on_capture(self):
        rgb, dep, intr = self.node.get_snapshot()
        if rgb is None or rgb.max() == 0:
            self._log("[エラー] カメラトピックのデータがまだ届いていません。少し待ってください。")
            return
        self._captured_rgb   = rgb
        self._captured_depth = dep
        self._intrinsics     = intr
        self._click_x = self._click_y = -1
        self._lbl_click.config(text="選択点: -")
        self._show_frame(rgb, overlay_text="FROZEN")
        if self._vis_win:
            self._vis_win.update_frozen(rgb)
        self._log(f"フレーム取得完了: shape={rgb.shape}  ─  クリックで物体を選択してください")
        self._set_state(S_CAPTURED)

    # ──────────────────────── 頭部制御 ─────────────────────────────────────────

    def _on_head_move(self):
        yaw   = self._neck_yaw_var.get()
        pitch = self._neck_pitch_var.get()
        self._on_neck_pitch_change()
        self._run_bg(self._do_head_move, yaw, pitch)

    def _do_head_move(self, yaw_deg: float, pitch_deg: float):
        if not _MOVEIT_OK:
            self._log("[頭部] MoveItPy 未利用可能 — スキップします")
            return
        robot     = self._get_robot()
        plan_p    = self._make_plan_params(robot, vel=0.3)
        head_comp = robot.get_planning_component(self.args.head_group)
        model     = robot.get_robot_model()
        rs        = RobotState(model)
        rs.set_joint_group_positions(self.args.head_group, [
            math.radians(yaw_deg),
            math.radians(pitch_deg),
        ])
        head_comp.set_start_state_to_current_state()
        head_comp.set_goal_state(robot_state=rs)
        _plan_and_execute(robot, head_comp, self._log, plan_p)
        self._log(f"[頭部] 移動完了: ヨー={yaw_deg:.1f}°, ピッチ={pitch_deg:.1f}°")

    def _on_head_home(self):
        self._run_bg(self._do_head_home)

    def _do_head_home(self):
        if not _MOVEIT_OK:
            self._log("[頭部] MoveItPy 未利用可能 — スキップします")
            return
        robot     = self._get_robot()
        plan_p    = self._make_plan_params(robot, vel=0.3)
        head_comp = robot.get_planning_component(self.args.head_group)
        head_comp.set_start_state_to_current_state()
        head_comp.set_goal_state(configuration_name="neck_init_pose")
        _plan_and_execute(robot, head_comp, self._log, plan_p)
        self._log("[頭部] 初期位置へ移動完了")

    # ──────────────────────── 姿勢推定 ─────────────────────────────────────────

    def _on_reselect(self):
        if self._captured_rgb is None:
            self._log("[再選択] フレームを先に取得してください")
            return
        self._click_x = self._click_y = -1
        self._lbl_click.config(text="選択点: -")
        self._show_frame(self._captured_rgb, overlay_text="FROZEN")
        if self._vis_win:
            self._vis_win.update_frozen(self._captured_rgb)
        self._log("別の物体を選択してください（画像をクリック）")
        self._set_state(S_CAPTURED)

    def _on_estimate(self):
        if self._click_x < 0:
            self._log("[エラー] 画像をクリックして物体を選択してください")
            return
        self._set_state(S_ESTIMATING)
        self._run_bg(self._do_estimate, on_error_state=S_SELECTED)

    def _do_estimate(self):
        if not _SERVER_OK:
            raise RuntimeError("サーバ通信モジュールが利用できません (requests/numpy を確認してください)")
        rgb    = self._captured_rgb
        dep_mm = self._captured_depth
        intr   = self._intrinsics
        depth_m = dep_mm.astype(np.float32) * self.args.depth_scale \
                  if dep_mm is not None else None

        import urllib.parse
        grasp_url = self.args.grasp_url or None
        if not grasp_url:
            parsed = urllib.parse.urlparse(self.args.server_url)
            grasp_url = f"{parsed.scheme}://{parsed.hostname}:8082"

        client = SAM6DClient(server_url=self.args.server_url,
                             timeout_mesh=300.0, timeout_pose=60.0,
                             grasp_server_url=grasp_url)
        self._sam6d_client = client
        self._log(f"[Step 1-3] server_grasp.py へ送信中 (選択点: {self._click_x}, {self._click_y})...")
        result = client.estimate_and_generate_grasp(
            rgb, depth_m, intr,
            click_x=self._click_x, click_y=self._click_y,
            gravity=self._gravity,
            mesh_method=self.args.mesh_method,
            num_samples=1,
        )
        R, t = result["R"], result["t"]
        self._R, self._t = R, t
        self._mesh_path = result.get("mesh_path", self.args.mesh_out)
        self._log(f"[Step 1-3完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        rgb_snap, intr_snap = rgb, intr
        self.root.after(0, lambda: self._vis_win and self._vis_win.update_pose(
            rgb_snap, R, t, intr_snap))

        grasp        = result["grasps"][0]
        mesh_scale_m = result["mesh_scale_m"]
        R_corr       = result["R_corr"].astype(np.float64)
        R_final      = (R.astype(np.float64) @ R_corr.T).astype(np.float32)
        pose         = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R_final)
        lh_cam       = normalized_to_camera(np.asarray(grasp["left_hand"]),  pose)
        rh_cam       = normalized_to_camera(np.asarray(grasp["right_hand"]), pose)
        self._finish_grasp(lh_cam, rh_cam)

    def _finish_grasp(self, lh_cam_all: np.ndarray, rh_cam_all: np.ndarray):
        lh_cam_all = np.asarray(lh_cam_all, dtype=np.float64)
        rh_cam_all = np.asarray(rh_cam_all, dtype=np.float64)
        lh_wrist = lh_cam_all[0] if lh_cam_all.ndim == 2 else lh_cam_all
        rh_wrist = rh_cam_all[0] if rh_cam_all.ndim == 2 else rh_cam_all
        self._lh_cam, self._rh_cam = lh_wrist, rh_wrist
        self._lh_cam_all = lh_cam_all if lh_cam_all.ndim == 2 else None
        self._rh_cam_all = rh_cam_all if rh_cam_all.ndim == 2 else None
        self._log(f"  左手首(cam): {lh_wrist}")
        self._log(f"  右手首(cam): {rh_wrist}")

        # ── パームフレーム → グリッパ回転角 ─────────────────────────────────
        self._log("[Step 4] パームフレームからグリッパ角を計算")
        l_euler = r_euler = None
        try:
            if self._lh_cam_all is not None and self._lh_cam_all.shape[0] >= 15:
                l_euler = _palm_to_gripper_euler(self._lh_cam_all, flip_z=False, r_home=_R_HOME_L)
                self._log(
                    f"  左グリッパ角(初期基準): yaw={l_euler[0]:.1f}°"
                    f"  pitch={l_euler[1]:.1f}°  roll={l_euler[2]:.1f}°"
                )
            if self._rh_cam_all is not None and self._rh_cam_all.shape[0] >= 15:
                r_euler = _palm_to_gripper_euler(self._rh_cam_all, flip_z=True, r_home=_R_HOME_R)
                self._log(
                    f"  右グリッパ角(base): yaw={r_euler[0]:.1f}°"
                    f"  pitch={r_euler[1]:.1f}°  roll={r_euler[2]:.1f}°"
                )
        except Exception as e:
            self._log(f"  [警告] グリッパ角計算失敗: {e}")

        self._log("[Step 5] TF2 変換: camera → base_link")
        x_off = self._x_offset_var.get()
        y_off = self._y_offset_var.get()
        z_off = self._z_offset_var.get()
        lh_base = self.node.camera_to_base(lh_wrist, self.args.camera_frame, self.args.base_frame,
                                            x_offset=x_off, y_offset=y_off, z_offset=z_off)
        rh_base = self.node.camera_to_base(rh_wrist, self.args.camera_frame, self.args.base_frame,
                                            x_offset=x_off, y_offset=y_off, z_offset=z_off)
        self._lh_base, self._rh_base = lh_base, rh_base
        rgb_snap, intr_snap = self._captured_rgb, self._intrinsics
        self.root.after(0, lambda: self._vis_win and self._vis_win.update_grasp(
            rgb_snap, lh_cam_all, rh_cam_all, intr_snap))
        self._log(f"  左手首(base): [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
        self._log(f"  右手首(base): [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
        self.node.publish_grasp_markers(lh_base, rh_base)
        self.node.publish_hand_keypoints_markers(
            self._lh_cam_all, self._rh_cam_all,
            self.args.camera_frame, self.args.base_frame,
            x_off, y_off, z_off,
        )

        # ゴール姿勢の中指方向マーカを表示
        try:
            lh_goal_xaxis = rh_goal_xaxis = None
            if self._lh_cam_all is not None and self._lh_cam_all.shape[0] >= 15:
                R_lh = _R_CAM_OPT_TO_BASE @ _compute_palm_frame(self._lh_cam_all, flip_z=False)
                lh_goal_xaxis = R_lh[:, 0]
            if self._rh_cam_all is not None and self._rh_cam_all.shape[0] >= 15:
                R_rh = _R_CAM_OPT_TO_BASE @ _compute_palm_frame(self._rh_cam_all, flip_z=True)
                rh_goal_xaxis = R_rh[:, 0]
            self._lh_goal_xaxis = lh_goal_xaxis
            self._rh_goal_xaxis = rh_goal_xaxis
            self.node.publish_goal_direction_markers(
                lh_base, rh_base,
                lh_xaxis=lh_goal_xaxis,
                rh_xaxis=rh_goal_xaxis,
            )
        except Exception as e:
            self._log(f"  [警告] ゴール中指方向マーカ失敗: {e}")

        if self._robot is not None:
            self.node.publish_joint_markers(self._robot)

        # j5/j6/j7 逆算: R_0→j4^{-1} @ R_goal を YZY 分解
        l_joints = self._palm_to_wrist_joints(
            self._lh_cam_all, flip_z=False, pre_j5_link='l_link4')
        r_joints = self._palm_to_wrist_joints(
            self._rh_cam_all, flip_z=True,  pre_j5_link='r_link4')
        if l_joints:
            self._log(f"  左 j5={l_joints[0]:.1f}°  j6={l_joints[1]:.1f}°  j7={l_joints[2]:.1f}°")
        if r_joints:
            self._log(f"  右 j5={r_joints[0]:.1f}°  j6={r_joints[1]:.1f}°  j7={r_joints[2]:.1f}°")

        _lj, _rj = l_joints, r_joints

        def _upd():
            self._lbl_lh.config(
                text=f"左手首: [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
            self._lbl_rh.config(
                text=f"右手首: [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
            if _lj is not None:
                self._l_roll_var.set(round(_lj[0], 1))   # j5 Roll
                self._l_pitch_var.set(round(_lj[1], 1))  # j6 Pitch
                self._l_yaw_var.set(round(_lj[2], 1))    # j7 Yaw
            if _rj is not None:
                self._r_roll_var.set(round(_rj[0], 1))
                self._r_pitch_var.set(round(_rj[1], 1))
                self._r_yaw_var.set(round(_rj[2], 1))
            self._set_state(S_DONE)
        self.root.after(0, _upd)

    # ──────────────────────── アーム移動 ───────────────────────────────────────

    def _on_move(self):
        if self._lh_base is None or self._rh_base is None:
            self._log("[エラー] 把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move, on_error_state=S_DONE)

    def _on_move_left(self):
        if self._lh_base is None:
            self._log("[エラー] 左手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move_left, on_error_state=S_DONE)

    def _on_move_right(self):
        if self._rh_base is None:
            self._log("[エラー] 右手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move_right, on_error_state=S_DONE)

    # ──────────────────────── 腰制御 ───────────────────────────────────────────

    def _on_waist_move(self):
        self._set_state(S_MOVING)
        self._run_bg(self._do_waist_move, on_error_state=S_DONE)

    def _on_waist_home(self):
        self._waist_angle_var.set(0.0)
        self._set_state(S_MOVING)
        self._run_bg(self._do_waist_home, on_error_state=S_DONE)

    def _do_waist_move(self):
        angle_rad = math.radians(self._waist_angle_var.get())
        self._log(f"[腰移動] {math.degrees(angle_rad):.1f} deg へ移動中...")
        robot = self._get_robot()
        traj_out = []
        ok = self.node.move_waist(robot, angle_rad, traj_out=traj_out)
        if ok:
            self._waist_angle_rad = angle_rad
            for t in traj_out:
                label = f"腰移動 {math.degrees(angle_rad):.1f} deg"
                self._plan_steps.append({"type": "waist", "label": label, "group": "waist_group", "trajectory": t})
            self._log(f"[腰移動完了] {math.degrees(angle_rad):.1f} deg — 計画ステップ数: {len(self._plan_steps)}")
        else:
            self._log("[腰移動失敗]")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_waist_home(self):
        self._log("[腰初期化] 0 deg へ移動中...")
        robot = self._get_robot()
        traj_out = []
        ok = self.node.move_waist(robot, 0.0, traj_out=traj_out)
        if ok:
            self._waist_angle_rad = 0.0
            for t in traj_out:
                self._plan_steps.append({"type": "waist", "label": "腰初期化 0.0 deg", "group": "waist_group", "trajectory": t})
            self._log(f"[腰初期化完了] — 計画ステップ数: {len(self._plan_steps)}")
        else:
            self._log("[腰初期化失敗]")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _auto_move_waist(self, robot):
        """auto チェックがONの場合、腰をスピンボックス角度に移動して _plan_steps に保存する。"""
        if not self._waist_auto_var.get():
            return
        angle_rad = math.radians(self._waist_angle_var.get())
        if abs(angle_rad - self._waist_angle_rad) < math.radians(0.5):
            return
        self._log(f"[腰自動移動] {math.degrees(angle_rad):.1f} deg...")
        traj_out = []
        ok = self.node.move_waist(robot, angle_rad, traj_out=traj_out)
        if ok:
            self._waist_angle_rad = angle_rad
            for t in traj_out:
                label = f"腰移動 {math.degrees(angle_rad):.1f} deg"
                self._plan_steps.append({"type": "waist", "label": label, "group": "waist_group", "trajectory": t})
        else:
            self._log("[警告] 腰自動移動失敗 — アーム移動を続行します")

    # ──────────────────────── アーム移動 ───────────────────────────────────────

    def _do_move(self):
        robot  = self._get_robot()
        self._auto_move_waist(robot)
        l_arm  = robot.get_planning_component(self.args.l_arm_group)
        r_arm  = robot.get_planning_component(self.args.r_arm_group)

        self._log("[移動] グラスプ位置へ移動中 (現在の向きを維持)...")
        traj_l, traj_r = [], []
        self.node.move_arm(robot, l_arm, self._make_plan_params(robot, vel=0.1), self._lh_base, self.args.l_pose_link, traj_out=traj_l)
        self.node.move_arm(robot, r_arm, self._make_plan_params(robot, vel=0.1), self._rh_base, self.args.r_pose_link, traj_out=traj_r)
        for t in traj_l:
            self._plan_steps.append({"type": "arm", "label": "左アームグラスプ位置へ移動", "group": self.args.l_arm_group, "trajectory": t})
        for t in traj_r:
            self._plan_steps.append({"type": "arm", "label": "右アームグラスプ位置へ移動", "group": self.args.r_arm_group, "trajectory": t})
        self._log(f"[移動完了] 計画ステップ数: {len(self._plan_steps)} — RViz で動作を確認してください")
        self.node.publish_joint_markers(robot)
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _move_single_arm(self, robot, side: str):
        self._auto_move_waist(robot)
        if side == "left":
            arm_group  = self.args.l_arm_group
            pose_link  = self.args.l_pose_link
            wrist_base = self._lh_base
        else:
            arm_group  = self.args.r_arm_group
            pose_link  = self.args.r_pose_link
            wrist_base = self._rh_base

        plan_p = self._make_plan_params(robot, vel=0.1)
        arm    = robot.get_planning_component(arm_group)

        self._log(f"[{side}アーム] グラスプ位置へ移動中 (現在の向きを維持)...")
        traj_out = []
        self.node.move_arm(robot, arm, plan_p, wrist_base, pose_link, traj_out=traj_out)
        label = "左アームグラスプ位置へ移動" if side == "left" else "右アームグラスプ位置へ移動"
        for t in traj_out:
            self._plan_steps.append({"type": "arm", "label": label, "group": arm_group, "trajectory": t})
        self._log(f"[{side}アーム完了] 計画ステップ数: {len(self._plan_steps)}")

    def _do_move_left(self):
        self._move_single_arm(self._get_robot(), "left")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_right(self):
        self._move_single_arm(self._get_robot(), "right")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _on_move_align_left(self):
        if self._lh_base is None:
            self._log("[エラー] 左手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move_align_left, on_error_state=S_DONE)

    def _on_move_align_right(self):
        if self._rh_base is None:
            self._log("[エラー] 右手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move_align_right, on_error_state=S_DONE)

    def _on_move_align_both(self):
        if self._lh_base is None or self._rh_base is None:
            self._log("[エラー] 把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_move_align_both, on_error_state=S_DONE)

    def _do_move_align_left(self):
        self._exec_move_and_align("left")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_align_right(self):
        self._exec_move_and_align("right")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_align_both(self):
        self._exec_move_and_align("left")
        time.sleep(0.3)
        self._exec_move_and_align("right")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _exec_move_and_align(self, side: str):
        """グラスプ位置へ移動しながら j6→j7 を GL/GR 方向に揃える。

        FK で現在の j6→j7 単位ベクトルを取得し Rotation.align_vectors で
        GL/GR への最小回転 rot_align を計算する。現在の EEF 姿勢行列に
        rot_align を左から掛けた R_new を目標姿勢として位置と同時に IK に渡す。
        """
        robot = self._get_robot()
        self._auto_move_waist(robot)
        plan_p = self._make_plan_params(robot, vel=0.1)

        if side == "left":
            arm_group  = self.args.l_arm_group
            pose_link  = self.args.l_pose_link
            wrist_base = self._lh_base
            goal_xaxis = self._lh_goal_xaxis
        else:
            arm_group  = self.args.r_arm_group
            pose_link  = self.args.r_pose_link
            wrist_base = self._rh_base
            goal_xaxis = self._rh_goal_xaxis

        if wrist_base is None or goal_xaxis is None:
            self._log(f"[{side}移動+回転] 把持座標またはターゲット方向データがありません")
            return

        # FK で現在の j6→j7 ベクトルと EEF 回転行列を取得
        link6_name = "l_link6" if side == "left" else "r_link6"
        with robot.get_planning_scene_monitor().read_only() as scene:
            rs = scene.current_state
            rs.update()
            tf6 = np.asarray(rs.get_global_link_transform(link6_name))
            tf7 = np.asarray(rs.get_global_link_transform(pose_link))

        vec = tf7[:3, 3] - tf6[:3, 3]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            self._log(f"[{side}移動+回転] j6→j7 ベクトルが零 — スキップ")
            return
        j6_to_j7 = vec / norm
        R7 = tf7[:3, :3]

        # j6→j7 方向を link7 フレームで表現（URDF 固有の定数ベクトル）
        target_dir = np.asarray(goal_xaxis, dtype=np.float64)
        target_dir = target_dir / np.linalg.norm(target_dir)
        t_local = R7.T @ j6_to_j7   # link7 フレームでの j6→j7 方向

        # t_local が target_dir を向くような基準 EEF 姿勢を構築
        # R_base @ t_local = target_dir となる R_base を求める
        rot_base, _ = Rotation.align_vectors([target_dir], [t_local])
        R_base = rot_base.as_matrix()

        dot = float(np.clip(np.dot(j6_to_j7, target_dir), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        self._log(
            f"[{side}移動+回転] j6→j7=({j6_to_j7[0]:+.3f},{j6_to_j7[1]:+.3f},{j6_to_j7[2]:+.3f}) "
            f"→ GL/GR=({target_dir[0]:+.3f},{target_dir[1]:+.3f},{target_dir[2]:+.3f}) "
            f"角度差={angle_deg:.1f}°  "
            f"目標位置: [{wrist_base[0]:.3f}, {wrist_base[1]:.3f}, {wrist_base[2]:.3f}]"
        )

        qx, qy, qz, qw = Rotation.from_matrix(R_base).as_quat()
        orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        arm = robot.get_planning_component(arm_group)
        traj_out = []
        ok = self.node.move_arm(robot, arm, plan_p, wrist_base, pose_link, orientation, traj_out=traj_out)
        if ok:
            label = f"{'左' if side == 'left' else '右'}アーム移動+回転(align)"
            for t in traj_out:
                self._plan_steps.append({"type": "arm", "label": label, "group": arm_group, "trajectory": t})
            self._log(f"[{side}移動+回転完了] 計画ステップ数: {len(self._plan_steps)}")
            self.node.publish_joint_markers(robot)
            try:
                lh_pos, lh_dir = self.node.read_link_direction(robot, "l_link6", self.args.l_pose_link)
                rh_pos, rh_dir = self.node.read_link_direction(robot, "r_link6", self.args.r_pose_link)
                self.node.publish_home_direction_markers(
                    lh_pos, rh_pos,
                    lh_xaxis=lh_dir,
                    rh_xaxis=rh_dir,
                )
                if self._lh_goal_xaxis is not None or self._rh_goal_xaxis is not None:
                    lh_base = self._lh_base if self._lh_base is not None else lh_pos
                    rh_base = self._rh_base if self._rh_base is not None else rh_pos
                    self.node.publish_goal_direction_markers(
                        lh_base, rh_base,
                        lh_xaxis=self._lh_goal_xaxis,
                        rh_xaxis=self._rh_goal_xaxis,
                    )
            except Exception as e:
                self.node.get_logger().warn(f"[marker] 方向マーカ更新失敗: {e}")
        else:
            self._log(f"[{side}移動+回転失敗]")

    def _on_rotate_left(self):
        if self._lh_base is None:
            self._log("[エラー] 左手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_rotate_arm, "left", on_error_state=S_DONE)

    def _on_rotate_right(self):
        if self._rh_base is None:
            self._log("[エラー] 右手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(self._do_rotate_arm, "right", on_error_state=S_DONE)

    def _palm_to_wrist_joints(self, joints_cam: np.ndarray, flip_z: bool,
                               pre_j5_link: str) -> "tuple[float, float, float] | None":
        """カメラ検出の23関節 → j5/j6/j7 絶対角 [deg] を FK 逆算で返す。

        R_j567 = R_0→j4^{-1} @ R_goal を YZY 分解して (j5, j6, j7) を得る。
        scipy は β(j6 中間角)を [0, π] で返すが、関節制限を超える場合は
        等価な別解 (α+π, −β, γ+π) に切り替える。
        pre_j5_link: j5 直前のリンク名 ('l_link4' / 'r_link4')
        robot 未初期化時は None を返す。
        """
        if self._robot is None:
            return None
        if joints_cam is None or np.asarray(joints_cam).shape[0] < 15:
            return None
        try:
            R_goal = _R_CAM_OPT_TO_BASE @ _compute_palm_frame(joints_cam, flip_z)
            with self._robot.get_planning_scene_monitor().read_only() as scene:
                rs = scene.current_state
                rs.update()
                tf_mat = np.asarray(rs.get_global_link_transform(pre_j5_link))
            R_0_to_4 = tf_mat[:3, :3]
            R_j567 = R_0_to_4.T @ R_goal

            # YZY 内因系分解: R = Ry(j5) @ Rz(j6) @ Ry(j7)
            # scipy は j6(中間角) を [0, π] で返す。
            j5, j6, j7 = Rotation.from_matrix(R_j567).as_euler('YZY')

            # 関節制限 [rad]
            # 左腕 j6: [-60°, +123°]  右腕 j6: [-123°, +60°]
            if pre_j5_link == 'l_link4':
                j5_lim = (-math.radians(90),  math.radians(90))
                j6_lim = (-math.radians(60),  math.radians(123))
                j7_lim = (-math.radians(170), math.radians(170))
            else:
                j5_lim = (-math.radians(90),  math.radians(90))
                j6_lim = (-math.radians(123), math.radians(60))
                j7_lim = (-math.radians(170), math.radians(170))

            def _in(v, lo, hi): return lo <= v <= hi
            def _norm(a): return ((a + math.pi) % (2 * math.pi)) - math.pi

            primary_ok = (_in(j5, *j5_lim) and _in(j6, *j6_lim) and _in(j7, *j7_lim))

            if not primary_ok:
                # 別解: Ry(α+π) @ Rz(−β) @ Ry(γ+π) は同一回転行列
                j5a = _norm(j5 + math.pi)
                j6a = -j6
                j7a = _norm(j7 + math.pi)
                alt_ok = (_in(j5a, *j5_lim) and _in(j6a, *j6_lim) and _in(j7a, *j7_lim))
                if alt_ok:
                    self._log(
                        f"  [逆算] 主解j6={math.degrees(j6):.0f}°が制限外 → "
                        f"別解j5={math.degrees(j5a):.0f}° j6={math.degrees(j6a):.0f}° j7={math.degrees(j7a):.0f}°"
                    )
                    j5, j6, j7 = j5a, j6a, j7a
                else:
                    self._log(
                        f"  [警告] 両YZY解が制限外: "
                        f"主({math.degrees(j5):.0f},{math.degrees(j6):.0f},{math.degrees(j7):.0f}) "
                        f"別({math.degrees(j5a):.0f},{math.degrees(j6a):.0f},{math.degrees(j7a):.0f})"
                    )

            # 丸め誤差検証
            R_check = R_0_to_4 @ Rotation.from_euler('YZY', [j5, j6, j7]).as_matrix()
            diff = float(np.max(np.abs(R_check - R_goal)))
            self._log(f"  [逆算検証] 残差={diff:.2e}  (0が正常)")

            return math.degrees(j5), math.degrees(j6), math.degrees(j7)
        except Exception as e:
            self._log(f"  [警告] j5/j6/j7 逆算失敗: {e}")
            return None

    def _do_rotate_arm(self, side: str):
        """現在の EEF 位置を保ちつつ GL/GR ゴール方向に EEF を向ける (Cartesian IK)。

        joints_cam から R_goal をその場で再計算し、現在位置 + R_goal 姿勢で
        move_arm を呼ぶ。関節角の直接指定ではないため関節制限の問題を MoveIt に委譲。
        """
        robot = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        if side == "left":
            arm_group = self.args.l_arm_group
            pose_link = self.args.l_pose_link
            joints_cam = self._lh_cam_all
            flip_z = False
        else:
            arm_group = self.args.r_arm_group
            pose_link = self.args.r_pose_link
            joints_cam = self._rh_cam_all
            flip_z = True

        # ゴール姿勢行列を取得 (把持姿勢データから再計算)
        if joints_cam is None or np.asarray(joints_cam).shape[0] < 15:
            self._log(f"[{side}アーム回転] 把持姿勢データがありません。先に姿勢推定を実行してください。")
            self.root.after(0, lambda: self._set_state(S_DONE))
            return

        R_goal = _R_CAM_OPT_TO_BASE @ _compute_palm_frame(joints_cam, flip_z)

        # 現在の EEF 位置を FK から取得 (位置は保持、姿勢のみ変更)
        eef_pos, _ = self.node.read_eef_pose(robot, pose_link)
        qx, qy, qz, qw = Rotation.from_matrix(R_goal).as_quat()
        orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

        self._log(
            f"[{side}アーム回転] 現在位置({eef_pos[0]:.3f},{eef_pos[1]:.3f},{eef_pos[2]:.3f})"
            f" → GL/GR方向へ Cartesian IK"
        )

        arm = robot.get_planning_component(arm_group)
        ok = self.node.move_arm(robot, arm, plan_p, eef_pos, pose_link, orientation)

        if ok:
            self._log(f"[{side}アーム回転完了]")
            self.node.publish_joint_markers(robot)
        else:
            self._log(f"[{side}アーム回転失敗] 現在位置で停止します")

        # 回転後のスライダ値を更新 (参考表示)
        try:
            pre_j5_link = 'l_link4' if side == 'left' else 'r_link4'
            joints_deg = self._palm_to_wrist_joints(joints_cam, flip_z, pre_j5_link)
            if joints_deg is not None:
                j5d, j6d, j7d = joints_deg
                if side == 'left':
                    self.root.after(0, lambda: (
                        self._l_roll_var.set(round(j5d, 1)),
                        self._l_pitch_var.set(round(j6d, 1)),
                        self._l_yaw_var.set(round(j7d, 1)),
                    ))
                else:
                    self.root.after(0, lambda: (
                        self._r_roll_var.set(round(j5d, 1)),
                        self._r_pitch_var.set(round(j6d, 1)),
                        self._r_yaw_var.set(round(j7d, 1)),
                    ))
        except Exception:
            pass

        # FK から両 j6→j7 方向を読んで現在方向マーカを更新
        try:
            lh_pos, lh_dir = self.node.read_link_direction(robot, "l_link6", self.args.l_pose_link)
            rh_pos, rh_dir = self.node.read_link_direction(robot, "r_link6", self.args.r_pose_link)
            self.node.publish_home_direction_markers(
                lh_pos, rh_pos,
                lh_xaxis=lh_dir,
                rh_xaxis=rh_dir,
            )
            if self._lh_goal_xaxis is not None or self._rh_goal_xaxis is not None:
                lh_base = self._lh_base if self._lh_base is not None else lh_pos
                rh_base = self._rh_base if self._rh_base is not None else rh_pos
                self.node.publish_goal_direction_markers(
                    lh_base, rh_base,
                    lh_xaxis=self._lh_goal_xaxis,
                    rh_xaxis=self._rh_goal_xaxis,
                )
        except Exception as e:
            self.node.get_logger().warn(f"[marker] 方向マーカ更新失敗: {e}")

        self.root.after(0, lambda: self._set_state(S_DONE))

    # ──────────────────────── グリッパ / 初期姿勢 ──────────────────────────────

    def _on_gripper_open(self):
        self._run_bg(self._do_gripper, "open")

    def _on_gripper_close(self):
        self._run_bg(self._do_gripper, "close")

    def _do_gripper(self, action: str):
        label    = "開放" if action == "open" else "閉鎖"
        robot    = self._get_robot()
        g_plan_p = self._make_plan_params(robot, vel=1.0)
        model    = robot.get_robot_model()
        angles   = (math.radians(-40), math.radians(40)) if action == "open" \
                   else (math.radians(-20), math.radians(20))
        self._log(f"グリッパ{label}中...")
        for comp_name, group, angle in (
            ("l_gripper_group", "l_gripper_group", angles[0]),
            ("r_gripper_group", "r_gripper_group", angles[1]),
        ):
            comp = robot.get_planning_component(comp_name)
            rs   = RobotState(model)
            rs.set_joint_group_positions(group, [angle])
            comp.set_start_state_to_current_state()
            comp.set_goal_state(robot_state=rs)
            _plan_and_execute(robot, comp, self._log, g_plan_p)

    def _on_recalc(self):
        if self._lh_cam is None or self._rh_cam is None:
            self._log("[再計算] まず姿勢推定を実行してください")
            return
        self._run_bg(self._do_recalc)

    def _do_recalc(self):
        x_off = self._x_offset_var.get()
        y_off = self._y_offset_var.get()
        z_off = self._z_offset_var.get()
        self._log(f"[再計算] X={x_off:+.3f}m  Y={y_off:+.3f}m  Z={z_off:+.3f}m")
        lh_base = self.node.camera_to_base(self._lh_cam, self.args.camera_frame,
                                            self.args.base_frame,
                                            x_offset=x_off, y_offset=y_off, z_offset=z_off)
        rh_base = self.node.camera_to_base(self._rh_cam, self.args.camera_frame,
                                            self.args.base_frame,
                                            x_offset=x_off, y_offset=y_off, z_offset=z_off)
        self._lh_base, self._rh_base = lh_base, rh_base
        self._log(f"  左手首(base): [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
        self._log(f"  右手首(base): [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
        self.node.publish_grasp_markers(lh_base, rh_base)
        self.node.publish_hand_keypoints_markers(
            self._lh_cam_all, self._rh_cam_all,
            self.args.camera_frame, self.args.base_frame,
            x_off, y_off, z_off,
        )
        if self._lh_goal_xaxis is not None or self._rh_goal_xaxis is not None:
            self.node.publish_goal_direction_markers(
                lh_base, rh_base,
                lh_xaxis=self._lh_goal_xaxis,
                rh_xaxis=self._rh_goal_xaxis,
            )
        self.root.after(0, lambda: (
            self._lbl_lh.config(
                text=f"左手首: [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]"),
            self._lbl_rh.config(
                text=f"右手首: [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]"),
        ))

    def _on_home(self):
        self._run_bg(self._do_home)

    def _on_home_left(self):
        self._run_bg(self._do_home_left)

    def _on_home_right(self):
        self._run_bg(self._do_home_right)

    def _on_joint_markers(self):
        robot = self._robot
        if robot is None:
            self._log("[エラー] MoveItPy 未初期化 — 起動ボタンを押してください")
            return
        self._run_bg(self._do_joint_markers)

    def _do_joint_markers(self):
        robot = self._get_robot()
        self.node.publish_joint_markers(robot)
        try:
            lh_pos, lh_dir = self.node.read_link_direction(robot, "l_link6", self.args.l_pose_link)
            rh_pos, rh_dir = self.node.read_link_direction(robot, "r_link6", self.args.r_pose_link)
            self.node.publish_home_direction_markers(
                lh_pos, rh_pos,
                lh_xaxis=lh_dir,
                rh_xaxis=rh_dir,
            )
            if self._lh_goal_xaxis is not None or self._rh_goal_xaxis is not None:
                lh_base = self._lh_base if self._lh_base is not None else lh_pos
                rh_base = self._rh_base if self._rh_base is not None else rh_pos
                self.node.publish_goal_direction_markers(
                    lh_base, rh_base,
                    lh_xaxis=self._lh_goal_xaxis,
                    rh_xaxis=self._rh_goal_xaxis,
                )
        except Exception as e:
            self.node.get_logger().warn(f"[marker] 方向マーカ更新失敗: {e}")

    def _do_home_left(self):
        if not _MOVEIT_OK:
            self._log("[初期姿勢] MoveItPy 未利用可能")
            return
        robot  = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        l_arm  = robot.get_planning_component(self.args.l_arm_group)
        self._log("[初期姿勢] 左アーム → l_arm_init_pose へ移動中...")
        l_arm.set_start_state_to_current_state()
        l_arm.set_goal_state(configuration_name="l_arm_init_pose")
        ok = _plan_and_execute(robot, l_arm, self._log, plan_p)
        if ok:
            self._log("[初期姿勢] 左アーム完了")
        else:
            self._log("[初期姿勢] 左アーム計画失敗")
        self.node.publish_joint_markers(robot)

    def _do_home_right(self):
        if not _MOVEIT_OK:
            self._log("[初期姿勢] MoveItPy 未利用可能")
            return
        robot  = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        r_arm  = robot.get_planning_component(self.args.r_arm_group)
        self._log("[初期姿勢] 右アーム → r_arm_init_pose へ移動中...")
        r_arm.set_start_state_to_current_state()
        r_arm.set_goal_state(configuration_name="r_arm_init_pose")
        ok = _plan_and_execute(robot, r_arm, self._log, plan_p)
        if ok:
            self._log("[初期姿勢] 右アーム完了")
        else:
            self._log("[初期姿勢] 右アーム計画失敗")
        self.node.publish_joint_markers(robot)

    def _do_home(self):
        if not _MOVEIT_OK:
            self._log("[初期姿勢] MoveItPy 未利用可能")
            return
        self._do_home_left()
        time.sleep(0.5)
        self._do_home_right()
        time.sleep(0.5)
        robot = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        waist  = robot.get_planning_component("waist_group")
        waist.set_start_state_to_current_state()
        waist.set_goal_state(configuration_name="waist_init_pose")
        _plan_and_execute(robot, waist, self._log, plan_p)
        self._waist_angle_rad = 0.0
        self._log("初期姿勢完了")
        try:
            robot = self._robot
            if robot is not None:
                lh_pos, lh_dir = self.node.read_link_direction(robot, "l_link6", self.args.l_pose_link)
                rh_pos, rh_dir = self.node.read_link_direction(robot, "r_link6", self.args.r_pose_link)
                self.node.publish_home_direction_markers(
                    lh_pos, rh_pos,
                    lh_xaxis=lh_dir,
                    rh_xaxis=rh_dir,
                )
        except Exception as e:
            self._log(f"[警告] 方向マーカ更新失敗: {e}")
            self.node.publish_home_direction_markers(_HOME_L, _HOME_R)

    # ──────────────────────── 計画保存 ─────────────────────────────────────────

    def _on_save_plan(self):
        import tkinter.messagebox as mb
        if not self._plan_steps:
            mb.showwarning("計画保存", "保存する計画がありません。\n移動ボタンで計画を実行してから保存してください。",
                           parent=self.root)
            self._log("[計画保存] 保存する計画がありません。移動ボタンで計画を実行してから保存してください。")
            return
        path = self._save_path_var.get().strip()
        if not path:
            mb.showwarning("計画保存", "保存先パスを入力してください。", parent=self.root)
            return
        self._log(f"[計画保存] {path} に保存中 ({len(self._plan_steps)} ステップ)...")
        self._run_bg(lambda: self._do_save_plan(path))

    def _do_save_plan(self, path: str):
        import pickle
        from rclpy.serialization import serialize_message
        data = []
        for step in self._plan_steps:
            s = dict(step)
            if s.get("type") in ("arm", "waist") and "trajectory" in s:
                s["trajectory"] = serialize_message(s["trajectory"])
            data.append(s)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self._log(f"[計画保存完了] {len(data)} ステップ → {path}")
        for i, s in enumerate(data):
            self._log(f"  Step {i+1}: [{s['type']}] {s['label']} ({s.get('group','-')})")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _on_clear_plan(self):
        self._plan_steps.clear()
        self._log("[計画クリア] 蓄積した計画ステップをリセットしました")

    # ──────────────────────── MoveItPy ヘルパー ────────────────────────────────

    def _get_robot(self):
        if not _MOVEIT_OK:
            raise RuntimeError("MoveItPy が利用できません (ROS2 環境を確認してください)")
        if self._robot is None:
            self._log("MoveItPy 初期化中 (数秒かかります)...")
            from ament_index_python.packages import get_package_share_directory
            from moveit_configs_utils import MoveItConfigsBuilder
            moveit_py_yaml = (
                get_package_share_directory("sciurus17_examples_py")
                + "/config/sciurus17_moveit_py_examples.yaml"
            )
            cfg = (
                MoveItConfigsBuilder("sciurus17")
                .planning_scene_monitor(
                    publish_robot_description=False,
                    publish_robot_description_semantic=False,
                )
                .moveit_cpp(file_path=moveit_py_yaml)
                .to_moveit_configs()
            )
            config_dict = cfg.to_dict()
            _remove_display_motion_path(config_dict)
            self._robot = MoveItPy(node_name="sciurus17_rviz_moveit",
                                   config_dict=config_dict)
            self._log("MoveItPy 初期化完了")
        return self._robot

    @staticmethod
    def _make_plan_params(robot, vel: float = 0.1):
        p = PlanRequestParameters(robot, "ompl_rrtc_default")
        p.max_velocity_scaling_factor     = vel
        p.max_acceleration_scaling_factor = vel
        p.planning_time                   = 10.0
        return p

    # ──────────────────────── 終了処理 ─────────────────────────────────────────

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.root.destroy()


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="sciurus17 把持コントロールパネル — RViz 仮想モード"
    )

    parser.add_argument("--rviz-image", default=None,
                        help="配信する保存 RGBD フォルダ (rgb.png / depth.png / cam.json)")
    parser.add_argument("--neck-pitch", type=float, default=0.0,
                        help="首ピッチ角の初期値 [deg]。正=下向き。重力方向計算に使用")

    parser.add_argument("--server-url",  default="http://10.40.1.126:8082",
                        help="server_grasp.py の URL (port 8082)")
    parser.add_argument("--grasp-url",   default="",
                        help="server_grasp.py の URL (省略時: --server-url と同じホスト:8082)")
    parser.add_argument("--mesh-out",    default="meshes/object.ply")
    parser.add_argument("--mesh-method", default="knn",
                        choices=["bpa", "poisson", "knn"])
    parser.add_argument("--depth-scale", type=float, default=0.001)

    parser.add_argument("--color-topic",
                        default="/head_camera/color/image_raw",
                        help="配信・受信するカラー画像トピック")
    parser.add_argument("--depth-topic",
                        default="/head_camera/aligned_depth_to_color/image_raw",
                        help="配信・受信する深度画像トピック")
    parser.add_argument("--info-topic",
                        default="/head_camera/color/camera_info",
                        help="配信・受信するカメラ情報トピック")

    parser.add_argument("--camera-frame", default="head_camera_color_optical_frame",
                        help="カメラの TF フレーム名 (ROS TF ツリー上の名前)")
    parser.add_argument("--base-frame",   default="base_link")
    parser.add_argument("--l-arm-group",  default="l_arm_group")
    parser.add_argument("--r-arm-group",  default="r_arm_group")
    parser.add_argument("--l-pose-link",  default="l_link7")
    parser.add_argument("--r-pose-link",  default="r_link7")
    parser.add_argument("--pre-grasp-z-offset", type=float, default=0.10)

    parser.add_argument("--head-group",       default="neck_group",
                        help="頭部 MoveIt planning group 名")
    parser.add_argument("--neck-yaw-joint",   default="neck_joint_1",
                        help="頭部ヨー関節名")
    parser.add_argument("--neck-pitch-joint", default="neck_joint_2",
                        help="頭部ピッチ関節名")

    parser.add_argument(
        "--launch-cmd",
        default=(
            'bash -c "source /ros2_ws/install/setup.bash && '
            'ros2 launch sciurus17_examples demo.launch.py '
            'use_mock_components:=true '
            'use_head_camera:=false use_chest_camera:=false"'
        ),
        help="sciurus17 起動コマンド (デフォルト: mock_components で RViz 起動)",
    )

    args = parser.parse_args()

    rclpy.init()
    node = RvizRobotNode(
        image_dir    = args.rviz_image,
        color_topic  = args.color_topic,
        depth_topic  = args.depth_topic,
        info_topic   = args.info_topic,
        camera_frame = args.camera_frame,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk(className="sciurus17")
    _app = SciurusRvizGUI(root, node, args)
    try:
        root.mainloop()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
