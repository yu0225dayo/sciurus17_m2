"""
ROS PC 側 RGBD HTTP サーバ + 手首座標 ROS パブリッシャー

役割:
    1. RealSense から RGBD を連続取得し、Client PC からの GET /rgbd に応答する
    2. Client PC から POST /wrist で受け取った手首座標を ROS トピックにパブリッシュする

依存:
    pip install flask pyrealsense2 opencv-python numpy
    ROS1 (rospy, geometry_msgs)

使用例:
    python rgbd_server.py --port 5000

ROS トピック:
    /wrist_left   geometry_msgs/PointStamped  左手首 (カメラ座標系 [m])
    /wrist_right  geometry_msgs/PointStamped  右手首 (カメラ座標系 [m])

rosbridge は不要。このスクリプト自体が ROS ノードとして動作する。
"""

import argparse
import base64
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from flask import Flask, jsonify, request
from geometry_msgs.msg import PointStamped

app = Flask(__name__)

# --------------------------------------------------------------------------
# グローバル状態
# --------------------------------------------------------------------------
_camera_lock = threading.Lock()
_latest: dict = {
    "rgb": None,
    "depth_mm": None,
    "intrinsics": None,
}
_ros_pubs: dict = {
    "left": None,
    "right": None,
}

# --------------------------------------------------------------------------
# カメラ
# --------------------------------------------------------------------------

def _camera_loop(width: int, height: int, fps: int):
    """バックグラウンドで RealSense を連続撮影する"""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    intrinsics = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "depth_scale": float(depth_scale),
    }

    align = rs.align(rs.stream.color)

    print("[Camera] ウォームアップ中 (30 フレーム)...")
    for _ in range(30):
        pipeline.wait_for_frames()
    print(f"[Camera] 起動完了 {width}x{height}@{fps}fps")

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        with _camera_lock:
            _latest["rgb"] = np.asanyarray(color_frame.get_data()).copy()
            _latest["depth_mm"] = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            _latest["intrinsics"] = intrinsics

    pipeline.stop()


# --------------------------------------------------------------------------
# HTTP エンドポイント
# --------------------------------------------------------------------------

@app.route("/rgbd", methods=["GET"])
def get_rgbd():
    """最新フレームを Base64 PNG で返す"""
    with _camera_lock:
        if _latest["rgb"] is None:
            return jsonify({"error": "カメラ未準備"}), 503
        rgb = _latest["rgb"].copy()
        depth_mm = _latest["depth_mm"].copy()
        intrinsics = _latest["intrinsics"].copy()

    _, rgb_buf = cv2.imencode(".png", rgb)
    _, dep_buf = cv2.imencode(".png", depth_mm)

    return jsonify({
        "rgb_b64":    base64.b64encode(rgb_buf.tobytes()).decode(),
        "depth_b64":  base64.b64encode(dep_buf.tobytes()).decode(),
        "intrinsics": intrinsics,
    })


@app.route("/wrist", methods=["POST"])
def receive_wrist():
    """
    Client PC から手首座標を受け取り ROS トピックにパブリッシュする

    Request JSON:
        {
            "left":  [x, y, z],   # 左手首 カメラ座標系 [m]
            "right": [x, y, z]    # 右手首 カメラ座標系 [m]
        }
    """
    data = request.get_json(force=True)
    stamp = rospy.Time.now()

    for side in ("left", "right"):
        if side not in data:
            continue
        xyz = data[side]
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_color_optical_frame"
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        _ros_pubs[side].publish(msg)
        print(f"[ROS] /wrist_{side} publish: "
              f"x={xyz[0]:.4f} y={xyz[1]:.4f} z={xyz[2]:.4f} [m]")

    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    ready = _latest["rgb"] is not None
    return jsonify({"status": "ok" if ready else "not_ready", "camera_ready": ready})


# --------------------------------------------------------------------------
# エントリポイント
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RGBD HTTP サーバ (ROS PC 側)")
    parser.add_argument("--port",   type=int, default=5000, help="HTTP ポート番号")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps",    type=int, default=30)
    args = parser.parse_args()

    # ROS ノード初期化
    rospy.init_node("rgbd_server", anonymous=True)
    _ros_pubs["left"]  = rospy.Publisher("/wrist_left",  PointStamped, queue_size=1)
    _ros_pubs["right"] = rospy.Publisher("/wrist_right", PointStamped, queue_size=1)
    print("[ROS] パブリッシャー準備完了: /wrist_left, /wrist_right")

    # カメラをバックグラウンドで起動
    cam_thread = threading.Thread(
        target=_camera_loop,
        args=(args.width, args.height, args.fps),
        daemon=True,
    )
    cam_thread.start()

    # Flask を別スレッドで起動 (rospy.spin() をメインスレッドで動かすため)
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()
    print(f"[HTTP] RGBD サーバ起動: http://0.0.0.0:{args.port}")

    rospy.spin()


if __name__ == "__main__":
    main()
