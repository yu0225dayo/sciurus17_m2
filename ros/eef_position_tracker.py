#!/usr/bin/env python3
"""
RViz インタラクティブマーカーでハンドを動かし到達可能空間を調査

マーカーをドラッグ → IK 計算 → 仮想ロボットがその位置に移動
届かない位置では NG と表示されロボットは動かない
Ctrl+C で /sciurus17_m2/eef_track.csv に保存して終了

RViz の設定:
  - InteractiveMarkers → Namespace: eef_marker
  - RobotModel → Description Topic: /robot_description
                  Robot State Topic: /display_robot_state
"""

import csv
import datetime
import sys
import threading
import tkinter as tk
from geometry_msgs.msg import Pose, Quaternion

try:
    import rclpy
    import rclpy.duration
    import rclpy.time
    from rclpy.node import Node
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from visualization_msgs.msg import (
        InteractiveMarker, InteractiveMarkerControl,
        InteractiveMarkerFeedback, Marker,
    )
    from moveit_msgs.msg import DisplayRobotState
    from tf2_ros import Buffer, TransformListener
except ImportError as e:
    print(f"[エラー] {e}", flush=True)
    sys.exit(1)

try:
    from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
    from moveit.planning import MoveItPy
    _MOVEIT_OK = True
except ImportError as e:
    print(f"[警告] MoveItPy 利用不可: {e}", flush=True)
    _MOVEIT_OK = False

CSV_PATH = "/sciurus17_m2/eef_track.csv"

ARMS = {
    "r_link7": {"group": "r_arm_group", "color": (1.0, 0.3, 0.3)},
}


class EefTracker(Node):
    def __init__(self):
        super().__init__("eef_position_tracker")
        self._server = InteractiveMarkerServer(self, "eef_marker")
        self._pub = self.create_publisher(DisplayRobotState, "/display_robot_state", 10)
        self._log: list[dict] = []
        self._lock = threading.Lock()
        self._robot = None
        self._fixed_orient: dict[str, "Quaternion | None"] = {"r_link7": None}
        self._last_poses: dict[str, "Pose"] = {}  # マーカー再表示用に最終位置を保持

        # TF
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # DDS ディスカバリ + robot_description トピック受信待機
        import time
        print("[初期化] DDS ディスカバリ待機中 (8秒)...", flush=True)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        # MoveItPy 初期化
        if _MOVEIT_OK:
            print("[初期化] MoveItPy 起動中...", flush=True)
            try:
                self._robot = MoveItPy(node_name="eef_tracker_moveit")
                print("[初期化] MoveItPy 完了", flush=True)
            except Exception as e:
                print(f"[警告] MoveItPy 初期化失敗: {e}", flush=True)

        # TF が届くまで待つ
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 起動時の表示姿勢: 前に習え（右腕のみ）
        init_pos = {"r_link7": self._get_eef_pos("r_link7")}  # TF fallback
        if self._robot is not None:
            try:
                import math
                model = self._robot.get_robot_model()
                rs = RobotState(model)
                rs.set_to_default_values()
                r_vals = [math.pi/2, -math.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0]
                rs.set_joint_group_positions("r_arm_group", r_vals)
                rs.update(True)
                display_msg = DisplayRobotState()
                display_msg.state = robotStateToRobotStateMsg(rs)
                self._pub.publish(display_msg)
                # FK からマーカー初期位置を取得
                eef_pose = rs.get_pose("r_link7")
                p = eef_pose.position
                init_pos["r_link7"] = [p.x, p.y, p.z]
                # CSV に記録
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[初期姿勢] R  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
                with self._lock:
                    self._log.append({
                        "timestamp": ts, "arm": "R", "ik": "INIT",
                        "x": f"{p.x:.4f}", "y": f"{p.y:.4f}", "z": f"{p.z:.4f}",
                    })
            except Exception as e:
                print(f"[警告] 初期姿勢設定失敗: {e}", flush=True)

        # マーカーを FK 位置に配置
        for name, cfg in ARMS.items():
            self._add_marker(name, init_pos[name], cfg["color"])

        self._server.applyChanges()

        print("=" * 50, flush=True)
        print("  マーカーをドラッグしてください", flush=True)
        print("  OK = IK 成功 (ロボットが移動)", flush=True)
        print("  NG = 届かない位置", flush=True)
        print("  Ctrl+C で終了 → CSV 保存", flush=True)
        print("=" * 50, flush=True)

    def _get_eef_pos(self, link: str) -> list:
        try:
            t = self._tf_buf.lookup_transform(
                "base_link", link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            tr = t.transform.translation
            print(f"[TF] {link}  X={tr.x:+.3f}  Y={tr.y:+.3f}  Z={tr.z:+.3f}", flush=True)
            return [tr.x, tr.y, tr.z]
        except Exception:
            default = [0.30, 0.20 if "l_" in link else -0.20, 0.30]
            print(f"[TF失敗] {link} → デフォルト {default}", flush=True)
            return default

    def _add_marker(self, name: str, pos: list, color: tuple):
        im = InteractiveMarker()
        im.header.frame_id = "base_link"
        im.name = name
        im.description = name
        im.scale = 0.15
        im.pose.position.x = float(pos[0])
        im.pose.position.y = float(pos[1])
        im.pose.position.z = float(pos[2])
        im.pose.orientation.w = 1.0

        r, g, b = color
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.06
        sphere.color.r = r
        sphere.color.g = g
        sphere.color.b = b
        sphere.color.a = 0.85

        # 球体（位置移動）
        move_ctrl = InteractiveMarkerControl()
        move_ctrl.name = "move_3d"
        move_ctrl.always_visible = True
        move_ctrl.interaction_mode = InteractiveMarkerControl.MOVE_3D
        move_ctrl.markers.append(sphere)
        im.controls.append(move_ctrl)

        # 回転リング（X/Y/Z 軸）
        for name, w, x, y, z in [
            ("rotate_x", 0.707, 0.707, 0.0,   0.0),
            ("rotate_z", 0.707, 0.0,   0.707, 0.0),
            ("rotate_y", 0.707, 0.0,   0.0,   0.707),
        ]:
            rot = InteractiveMarkerControl()
            rot.name = name
            rot.orientation.w = w
            rot.orientation.x = x
            rot.orientation.y = y
            rot.orientation.z = z
            rot.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            im.controls.append(rot)

        self._server.insert(im, feedback_callback=self._on_feedback)
        self._last_poses[name] = im.pose  # 初期位置を保存

    def hide_marker(self, link: str):
        marker = self._server.get(link)
        if marker:
            self._last_poses[link] = marker.pose
        self._server.erase(link)
        self._server.applyChanges()

    def show_marker(self, link: str):
        last = self._last_poses.get(link)
        pos = [last.position.x, last.position.y, last.position.z] if last else [0.3, -0.2, 0.3]
        self._add_marker(link, pos, ARMS[link]["color"])
        if last:
            self._server.setPose(link, last)
        self._server.applyChanges()

    def _ik_and_collision(self, group: str, link: str, pose, timeout: float):
        """IK + 自己衝突チェック。(ok, rs) を返す。"""
        model = self._robot.get_robot_model()
        rs = RobotState(model)
        rs.set_to_default_values()
        if not rs.set_from_ik(group, pose, link, timeout):
            return False, rs
        rs.update(True)
        try:
            with self._robot.get_planning_scene_monitor().read_only() as scene:
                valid = scene.is_state_valid(rs, group)
        except Exception as e:
            print(f"[警告] 衝突チェック失敗 (IK結果を使用): {e}", flush=True)
            valid = True
        return valid, rs

    def _on_feedback(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return
        self._last_poses[feedback.marker_name] = feedback.pose

        p = feedback.pose.position
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        label = "L" if feedback.marker_name == "l_link7" else "R"

        if self._robot is None:
            print(f"[{ts}] {label}  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
            with self._lock:
                self._log.append({
                    "timestamp": ts, "arm": label, "ik": "N/A",
                    "x": f"{p.x:.4f}", "y": f"{p.y:.4f}", "z": f"{p.z:.4f}",
                })
            return

        group = ARMS[feedback.marker_name]["group"]
        try:
            pose = feedback.pose
            fixed = self._fixed_orient.get(feedback.marker_name)
            if fixed is not None:
                pose = type(pose)()
                pose.position = feedback.pose.position
                pose.orientation = fixed
            ok, rs = self._ik_and_collision(group, feedback.marker_name, pose, 0.05)
            if ok:
                display_msg = DisplayRobotState()
                display_msg.state = robotStateToRobotStateMsg(rs)
                self._pub.publish(display_msg)
                print(f"[{ts}] {label} OK  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
                with self._lock:
                    self._log.append({
                        "timestamp": ts, "arm": label, "ik": "OK",
                        "x": f"{p.x:.4f}", "y": f"{p.y:.4f}", "z": f"{p.z:.4f}",
                    })
            else:
                print(f"[{ts}] {label} NG  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
        except Exception as e:
            print(f"[エラー] IK: {e}", flush=True)

    def compute_ik(self, link: str, pose):
        """キー操作から直接 IK を呼ぶ用。"""
        p = pose.position
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        label = "L" if link == "l_link7" else "R"
        if self._robot is None:
            print(f"[{ts}] {label}  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
            return
        group = ARMS[link]["group"]
        try:
            ik_pose = Pose()
            ik_pose.position = pose.position
            fixed = self._fixed_orient.get(link)
            ik_pose.orientation = fixed if fixed is not None else pose.orientation
            ok, rs = self._ik_and_collision(group, link, ik_pose, 0.05)
            if ok:
                display_msg = DisplayRobotState()
                display_msg.state = robotStateToRobotStateMsg(rs)
                self._pub.publish(display_msg)
                print(f"[{ts}] {label} OK  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
                with self._lock:
                    self._log.append({"timestamp": ts, "arm": label, "ik": "OK",
                                      "x": f"{p.x:.4f}", "y": f"{p.y:.4f}", "z": f"{p.z:.4f}"})
            else:
                print(f"[{ts}] {label} NG  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}", flush=True)
        except Exception as e:
            print(f"[エラー] IK: {e}", flush=True)

    def sweep(self, link: str, step: float, done_cb=None):
        """X[0,0.5] Y[-0.5,0.5] Z[0,0.5] をグリッドスキャンして OK/NG を記録。"""
        import numpy as np
        xs = [round(v, 4) for v in list(np.arange(0.0, 0.5 + step * 0.5, step))]
        ys = [round(v, 4) for v in list(np.arange(-0.5, 0.5 + step * 0.5, step))]
        zs = [round(v, 4) for v in list(np.arange(0.0, 0.5 + step * 0.5, step))]
        total = len(xs) * len(ys) * len(zs)
        label = "L" if link == "l_link7" else "R"
        group = ARMS[link]["group"]
        fixed = self._fixed_orient.get(link)
        print(f"[スキャン開始] {label}  格子点数={total}  step={step}m", flush=True)
        if fixed:
            q = fixed
            print(f"[向き] w={q.w:.3f} x={q.x:.3f} y={q.y:.3f} z={q.z:.3f}", flush=True)
        else:
            print("[向き] デフォルト (w=1)", flush=True)

        ok_count = 0
        ng_count = 0
        for i, x in enumerate(xs):
            for y in ys:
                for z in zs:
                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    if fixed is not None:
                        pose.orientation = fixed
                    else:
                        pose.orientation.w = 1.0
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    try:
                        ok, rs = self._ik_and_collision(group, link, pose, 0.02)
                    except Exception as e:
                        print(f"[エラー] IK: {e}", flush=True)
                        ok = False
                    ik_str = "OK" if ok else "NG"
                    if ok:
                        ok_count += 1
                        display_msg = DisplayRobotState()
                        display_msg.state = robotStateToRobotStateMsg(rs)
                        self._pub.publish(display_msg)
                    else:
                        ng_count += 1
                    with self._lock:
                        self._log.append({
                            "timestamp": ts, "arm": label, "ik": ik_str,
                            "x": f"{x:.4f}", "y": f"{y:.4f}", "z": f"{z:.4f}",
                        })
            done = (i + 1) * len(ys) * len(zs)
            print(f"[スキャン] {done}/{total}  OK={ok_count}  NG={ng_count}", flush=True)

        print(f"[スキャン完了] {label}  OK={ok_count}/{total}", flush=True)
        if done_cb:
            done_cb()

    def save(self):
        if not self._log:
            print("[CSV] 記録なし", flush=True)
            return
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "arm", "x", "y", "z", "ik"])
            w.writeheader()
            w.writerows(self._log)
        print(f"[CSV保存] {len(self._log)} 件 → {CSV_PATH}", flush=True)


class ControlGui:
    STEP = 0.01  # 1キー押しあたりの移動量 [m]

    # key → (dx, dy, dz)
    KEY_MAP = {
        "w": ( 1,  0,  0), "s": (-1,  0,  0),  # X 軸
        "a": ( 0,  1,  0), "d": ( 0, -1,  0),  # Y 軸
        "q": ( 0,  0,  1), "e": ( 0,  0, -1),  # Z 軸
    }

    def __init__(self, node: "EefTracker"):
        self._node = node
        self.root = tk.Tk()
        self.root.title("EEF Tracker")
        self.root.resizable(False, False)
        self._btns: dict[str, tk.Button] = {}

        # ── キー操作説明 ──
        tk.Label(self.root, text="右手: W/S=X前後  A/D=Y左右  Q/E=Z上下",
                 font=("Courier", 8), fg="#666666").pack(padx=10, pady=(8, 8))

        # ── 向き固定ボタン ──
        tk.Label(self.root, text="向き固定", font=("Helvetica", 9, "bold")
                 ).pack(anchor=tk.W, padx=10)
        b = tk.Button(
            self.root, text="右手 向き固定", width=20,
            bg="#5a2a2a", fg="white", font=("Helvetica", 9),
            command=lambda: self._toggle("r_link7"),
        )
        b.pack(padx=10, pady=3)
        self._btns["r_link7"] = b

        # ── グリッドスキャン ──
        tk.Label(self.root, text="可動域グリッドスキャン", font=("Helvetica", 9, "bold")
                 ).pack(anchor=tk.W, padx=10, pady=(10, 0))
        tk.Label(self.root, text="X:0~0.5  Y:-0.5~0.5  Z:0~0.5 [m]",
                 font=("Courier", 8), fg="#555555").pack(padx=10)
        step_frame = tk.Frame(self.root)
        step_frame.pack(padx=10, pady=2, anchor=tk.W)
        tk.Label(step_frame, text="ステップ [m]:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._step_var = tk.StringVar(value="0.05")
        tk.Entry(step_frame, textvariable=self._step_var, width=6,
                 font=("Courier", 9)).pack(side=tk.LEFT, padx=4)
        self._scan_btns: dict[str, tk.Button] = {}
        b = tk.Button(
            self.root, text="右手 スキャン", width=20,
            bg="#6a1a1a", fg="white", font=("Helvetica", 9),
            command=lambda: self._start_scan("r_link7"),
        )
        b.pack(padx=10, pady=2)
        self._scan_btns["r_link7"] = b

        # キーバインド（ウィンドウにフォーカスが必要）
        for key in self.KEY_MAP:
            self.root.bind(f"<{key}>", self._on_key)
            self.root.bind(f"<{key.upper()}>", self._on_key)

        self.root.bind("<FocusIn>", lambda e: self.root.config(bg="#e8f4e8"))
        self.root.bind("<FocusOut>", lambda e: self.root.config(bg="#f0f0f0"))

    def _start_scan(self, link: str):
        try:
            step = float(self._step_var.get())
        except ValueError:
            print("[エラー] ステップ値が不正です", flush=True)
            return
        for b in self._scan_btns.values():
            b.config(state=tk.DISABLED)
        for b in self._btns.values():
            b.config(state=tk.DISABLED)
        self._node.hide_marker(link)
        self._scan_link = link

        def _done():
            self.root.after(0, self._on_scan_done)

        threading.Thread(
            target=self._node.sweep, args=(link, step), kwargs={"done_cb": _done},
            daemon=True,
        ).start()

    def _on_scan_done(self):
        self._node.show_marker(self._scan_link)
        for b in self._scan_btns.values():
            b.config(state=tk.NORMAL)
        for b in self._btns.values():
            b.config(state=tk.NORMAL)
        self._node.save()
        print("[スキャン] CSV 保存完了", flush=True)

    def _on_key(self, event: tk.Event):
        key = event.char.lower()
        delta = self.KEY_MAP.get(key)
        if delta is None:
            return
        link = "r_link7"
        marker = self._node._server.get(link)
        if marker is None:
            return
        pose = marker.pose
        pose.position.x += delta[0] * self.STEP
        pose.position.y += delta[1] * self.STEP
        pose.position.z += delta[2] * self.STEP
        self._node._server.setPose(link, pose)
        self._node._server.applyChanges()
        # IK を手動トリガー
        self._node.compute_ik(link, pose)

    def _toggle(self, link: str):
        if self._node._fixed_orient[link] is None:
            marker = self._node._server.get(link)
            if marker is None:
                return
            self._node._fixed_orient[link] = marker.pose.orientation
            self._btns[link].config(bg="#aa6600",
                text=self._btns[link]["text"].replace("固定", "固定中 (解除)"))
            print(f"[向き固定] {link}", flush=True)
        else:
            self._node._fixed_orient[link] = None
            orig = {"l_link7": "#2a5a3a", "r_link7": "#5a2a2a"}[link]
            self._btns[link].config(bg=orig,
                text=self._btns[link]["text"].replace("固定中 (解除)", "固定"))
            print(f"[向き解除] {link}", flush=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    rclpy.init()
    node = EefTracker()
    gui = ControlGui(node)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        rclpy.shutdown()
