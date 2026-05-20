#!/usr/bin/env python3
"""
sciurus17 把持コントロールパネル (tkinter GUI) — RViz 仮想モード

ROS2 + MoveIt2 必須 / 実機ハードウェア不要。
demo.launch.py (use_mock_components:=true) で RViz を起動し、
保存済み RGBD を ROSカメラトピックとして配信してカメラストリームを再現する。
把持計画は MoveItPy で実行され、仮想ロボットの軌道が RViz で可視化される。

起動 (sciurus17/ ディレクトリで):
  docker compose run --rm sciurus17 \\
      python3 /robo_ros/ros/sciurus17_gui_rviz.py \\
      --rviz-image /robo_ros/client/saved_data/test_20260422_200416 \\
      --server-url http://10.40.1.126:8082

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

def _gravity_from_neck_pitch(neck_pitch_deg: float) -> np.ndarray:
    theta = math.radians(neck_pitch_deg)
    return np.array([0.0, -math.cos(theta), -math.sin(theta)], dtype=np.float32)


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


def _plan_and_execute(robot, comp, log_fn, params) -> bool:
    result = comp.plan(single_plan_parameters=params)
    if result:
        robot.execute(result.trajectory, controllers=[])
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
        # 起動後 1s で DELETEALL を送り、RViz がトピックを認識できるようにする
        self._marker_init_timer = self.create_timer(1.0, self._init_markers_once)

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

    def move_arm(self, robot, arm_comp, params, xyz_base, pose_link: str,
                 orientation) -> bool:
        """MoveItPy で目標位置へ移動。軌道は RViz で可視化される。"""
        goal = PoseStamped()
        goal.header.frame_id = "base_link"
        goal.pose = Pose(
            position=Point(x=float(xyz_base[0]),
                           y=float(xyz_base[1]),
                           z=float(xyz_base[2])),
            orientation=orientation,
        )
        arm_comp.set_start_state_to_current_state()
        arm_comp.set_goal_state(pose_stamped_msg=goal, pose_link=pose_link)
        return _plan_and_execute(robot, arm_comp, self.get_logger().error, params)


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

        # 4. アーム制御
        f5 = section("4. アーム制御")
        add_btn(f5, "▶  両アームを移動",   self._on_move,          "move",       color="#3d5b7a")
        add_btn(f5, "▶  左アームのみ",     self._on_move_left,     "move_left",  color="#3a547a")
        add_btn(f5, "▶  右アームのみ",     self._on_move_right,    "move_right", color="#3a547a")
        add_btn(f5, "○  グリッパ 開",     self._on_gripper_open,  "g_open")
        add_btn(f5, "●  グリッパ 閉",     self._on_gripper_close, "g_close")
        add_btn(f5, "⌂  初期姿勢へ",      self._on_home,          "home")

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
            "move":        s == S_DONE and not w,
            "log_btn":     True,
            "move_left":   s == S_DONE and not w,
            "move_right":  s == S_DONE and not w,
            "g_open":      can_arm,
            "g_close":     can_arm,
            "home":        can_arm,
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
                for line in self._process.stdout:
                    self._log(f"[ROS] {line.rstrip()}")
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
        self._log(f"  左手首(cam): {lh_wrist}")
        self._log(f"  右手首(cam): {rh_wrist}")
        self._log("[Step 4] TF2 変換: camera → base_link")
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

        def _upd():
            self._lbl_lh.config(
                text=f"左手首: [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
            self._lbl_rh.config(
                text=f"右手首: [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
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

    def _do_move(self):
        robot  = self._get_robot()
        l_arm  = robot.get_planning_component(self.args.l_arm_group)
        r_arm  = robot.get_planning_component(self.args.r_arm_group)
        plan_p = self._make_plan_params(robot, vel=0.1)
        q_l = Quaternion(x=-0.707, y=0.0, z=0.0, w=0.707)
        q_r = Quaternion(x=0.707,  y=0.0, z=0.0, w=0.707)

        Z = self.args.pre_grasp_z_offset
        lh_pre = self._lh_base.copy(); lh_pre[2] += Z
        rh_pre = self._rh_base.copy(); rh_pre[2] += Z
        self._log(f"[移動] プリグラスプへ (z+{Z:.2f}m)...")
        self.node.move_arm(robot, l_arm, plan_p, lh_pre, self.args.l_pose_link, q_l)
        self.node.move_arm(robot, r_arm, plan_p, rh_pre, self.args.r_pose_link, q_r)

        self._log("[移動] グラスプ位置へ下降...")
        self.node.move_arm(robot, l_arm, plan_p, self._lh_base, self.args.l_pose_link, q_l)
        self.node.move_arm(robot, r_arm, plan_p, self._rh_base, self.args.r_pose_link, q_r)
        self._log("[移動完了] RViz で動作を確認してください")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _move_single_arm(self, robot, side: str):
        if side == "left":
            arm_group   = self.args.l_arm_group
            pose_link   = self.args.l_pose_link
            orientation = Quaternion(x=-0.707, y=0.0, z=0.0, w=0.707)
            wrist_base  = self._lh_base
        else:
            arm_group   = self.args.r_arm_group
            pose_link   = self.args.r_pose_link
            orientation = Quaternion(x=0.707, y=0.0, z=0.0, w=0.707)
            wrist_base  = self._rh_base

        plan_p = self._make_plan_params(robot, vel=0.1)
        arm    = robot.get_planning_component(arm_group)

        Z = self.args.pre_grasp_z_offset
        pre = wrist_base.copy(); pre[2] += Z
        self._log(f"[{side}アーム] プリグラスプへ移動 (z+{Z:.2f}m)...")
        self.node.move_arm(robot, arm, plan_p, pre, pose_link, orientation)

        self._log(f"[{side}アーム] グラスプ位置へ下降...")
        self.node.move_arm(robot, arm, plan_p, wrist_base, pose_link, orientation)
        self._log(f"[{side}アーム完了]")

    def _do_move_left(self):
        self._move_single_arm(self._get_robot(), "left")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_right(self):
        self._move_single_arm(self._get_robot(), "right")
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
        self.root.after(0, lambda: (
            self._lbl_lh.config(
                text=f"左手首: [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]"),
            self._lbl_rh.config(
                text=f"右手首: [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]"),
        ))

    def _on_home(self):
        self._run_bg(self._do_home)

    def _do_home(self):
        if not _MOVEIT_OK:
            self._log("[初期姿勢] MoveItPy 未利用可能")
            return
        robot  = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        self._log("初期姿勢へ移動中...")
        for arm, name in (
            (robot.get_planning_component(self.args.l_arm_group), "l_arm_init_pose"),
            (robot.get_planning_component(self.args.r_arm_group), "r_arm_init_pose"),
        ):
            arm.set_start_state_to_current_state()
            arm.set_goal_state(configuration_name=name)
            _plan_and_execute(robot, arm, self._log, plan_p)
        self._log("初期姿勢完了")

    # ──────────────────────── MoveItPy ヘルパー ────────────────────────────────

    def _get_robot(self):
        if not _MOVEIT_OK:
            raise RuntimeError("MoveItPy が利用できません (ROS2 環境を確認してください)")
        first_init = self._robot is None
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
                    publish_robot_description=True,
                    publish_robot_description_semantic=True,
                )
                .moveit_cpp(file_path=moveit_py_yaml)
                .to_moveit_configs()
            )
            self._robot = MoveItPy(node_name="sciurus17_rviz_moveit",
                                   config_dict=cfg.to_dict())
            self._log("MoveItPy 初期化完了")
        if first_init:
            self._log("[自動] 起動後 初期姿勢へ移動中 (l_arm_init_pose / r_arm_init_pose)...")
            plan_p = self._make_plan_params(self._robot, vel=0.3)
            for arm, name in (
                (self._robot.get_planning_component(self.args.l_arm_group), "l_arm_init_pose"),
                (self._robot.get_planning_component(self.args.r_arm_group), "r_arm_init_pose"),
            ):
                arm.set_start_state_to_current_state()
                arm.set_goal_state(configuration_name=name)
                _plan_and_execute(self._robot, arm, self._log, plan_p)
            self._log("[自動] 初期姿勢完了")
        return self._robot

    @staticmethod
    def _make_plan_params(robot, vel: float = 0.1):
        p = PlanRequestParameters(robot, "ompl_rrtc_default")
        p.max_velocity_scaling_factor     = vel
        p.max_acceleration_scaling_factor = vel
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
