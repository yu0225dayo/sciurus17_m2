#!/usr/bin/env python3
"""
sciurus17 可動域調査 — 仮想空間 (RViz) 版

起動手順:
  1. demo.launch.py (use_mock_components:=true) で RViz を起動
  2. 本スクリプトを実行:
       python3 /sciurus17_m2/ros/sciurus17_range_rviz.py
  3. [関節制限を表示] → URDF の可動域を確認
  4. [関節スイープ] → 1軸ずつ動かして RViz で軌跡を確認
  5. [ランダムサンプリング] → 到達可能空間を /workspace_markers で可視化
  6. [CSV保存] で結果を書き出し

RViz 設定:
  - MarkerArray トピック /workspace_markers を追加すると到達可能点群が表示されます

依存:
  ROS2 Jazzy + MoveIt2 + sciurus17_ros (demo.launch.py 起動済み)
"""

import csv
import datetime
import math
import random
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    from geometry_msgs.msg import Point
except ImportError as e:
    print(f"[エラー] ROS2 が見つかりません: {e}")
    sys.exit(1)

_MOVEIT_OK = False
try:
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy, PlanRequestParameters
    _MOVEIT_OK = True
except ImportError:
    pass

# ── 定数 ───────────────────────────────────────────────────────────────────────
ARM_GROUPS = {
    "左アーム (l_arm_group)": "l_arm_group",
    "右アーム (r_arm_group)": "r_arm_group",
}
EEF_LINK_MAP = {
    "l_arm_group": "l_link7",
    "r_arm_group": "r_link7",
}

BG = "#2b2b2b"; PANEL_BG = "#3c3f41"; BTN_BG = "#555759"; BTN_FG = "white"
LOG_BG = "#1e1e1e"; LOG_FG = "#cccccc"


# ── ROS2 ノード ────────────────────────────────────────────────────────────────
class WorkspaceNode(Node):
    def __init__(self):
        super().__init__("sciurus17_range_rviz")
        self._pub = self.create_publisher(MarkerArray, "/workspace_markers", 10)
        self._robot: "MoveItPy | None" = None
        self._marker_id = 0
        self._stop_flag = False

    def get_robot(self) -> "MoveItPy":
        if self._robot is None:
            if not _MOVEIT_OK:
                raise RuntimeError("MoveItPy が利用できません")
            self._robot = MoveItPy(node_name="sciurus17_range_rviz")
        return self._robot

    def get_joint_limits(self, group_name: str) -> list[tuple[str, float, float]]:
        """グループの (関節名, min_rad, max_rad) リストを返す。"""
        robot   = self.get_robot()
        model   = robot.get_robot_model()
        jmg     = model.get_joint_model_group(group_name)
        joints  = jmg.get_active_joint_model_names()
        result  = []
        for jn in joints:
            jm     = model.get_joint_model(jn)
            bounds = jm.get_variable_bounds()
            lo = bounds[0].min_position if bounds else -math.pi
            hi = bounds[0].max_position if bounds else  math.pi
            result.append((jn, lo, hi))
        return result

    def fk_eef(self, group_name: str, joint_angles: list[float]) -> "np.ndarray | None":
        """関節角度から EEF (l_link7 / r_link7) の XYZ を返す。"""
        robot    = self.get_robot()
        model    = robot.get_robot_model()
        eef_link = EEF_LINK_MAP.get(group_name, "l_link7")
        try:
            with robot.get_planning_scene_monitor().read_only() as scene:
                rs = scene.current_state
                rs.set_joint_group_positions(group_name, joint_angles)
                rs.update()
                tf = rs.get_global_link_transform(eef_link)
                return np.array([float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3])])
        except Exception:
            return None

    def sweep_joint(
        self,
        group_name: str,
        joint_index: int,
        n_steps: int,
        vel: float,
        log_fn,
    ) -> list[np.ndarray]:
        """指定関節を最小〜最大に N ステップでスイープし、FK 結果のリストを返す。"""
        robot   = self.get_robot()
        model   = robot.get_robot_model()
        limits  = self.get_joint_limits(group_name)
        jname, lo, hi = limits[joint_index]
        jmg     = model.get_joint_model_group(group_name)
        all_jnames = list(jmg.get_active_joint_model_names())

        log_fn(f"[スイープ] {jname}  {math.degrees(lo):.1f}° → {math.degrees(hi):.1f}°  {n_steps}ステップ")

        eef_link = EEF_LINK_MAP.get(group_name, "l_link7")
        comp     = robot.get_planning_component(group_name)
        plan_p   = PlanRequestParameters(robot, "ompl_rrtc")
        plan_p.max_velocity_scaling_factor    = vel
        plan_p.max_acceleration_scaling_factor = vel

        positions: list[np.ndarray] = []
        self._stop_flag = False

        for step in range(n_steps + 1):
            if self._stop_flag:
                log_fn("[スイープ] 中断しました")
                break
            angle = lo + (hi - lo) * step / n_steps

            # 現在状態を取得して対象関節だけ変更
            with robot.get_planning_scene_monitor().read_only() as scene:
                rs_cur = scene.current_state
                cur_angles = list(rs_cur.get_joint_group_positions(group_name))

            cur_angles[joint_index] = angle

            # FK で位置を記録
            pos = self.fk_eef(group_name, cur_angles)
            if pos is not None:
                positions.append(pos)

            # MoveItPy で実際に移動
            rs_goal = RobotState(model)
            rs_goal.set_joint_group_positions(group_name, cur_angles)
            comp.set_start_state_to_current_state()
            comp.set_goal_state(robot_state=rs_goal)
            result = comp.plan(single_plan_parameters=plan_p)
            if result:
                robot.execute(result.trajectory, controllers=[])
                log_fn(f"  step {step+1}/{n_steps+1}: {jname}={math.degrees(angle):+.1f}°  "
                       f"EEF=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})" if pos is not None
                       else f"  step {step+1}/{n_steps+1}: 計画成功 (FK失敗)")
            else:
                log_fn(f"  step {step+1}/{n_steps+1}: 計画失敗 → スキップ")

        return positions

    def sample_workspace(
        self,
        group_name: str,
        n_samples: int,
        log_fn,
    ) -> list[np.ndarray]:
        """ランダム関節サンプリングで到達可能空間を収集する (ロボットは動かさない)。"""
        limits   = self.get_joint_limits(group_name)
        eef_link = EEF_LINK_MAP.get(group_name, "l_link7")
        robot    = self.get_robot()
        model    = robot.get_robot_model()

        log_fn(f"[サンプリング] {group_name}  {n_samples} サンプル開始")
        positions: list[np.ndarray] = []
        self._stop_flag = False

        for i in range(n_samples):
            if self._stop_flag:
                log_fn(f"[サンプリング] 中断 ({i}/{n_samples} 完了)")
                break

            angles = [random.uniform(lo, hi) for _, lo, hi in limits]
            try:
                with robot.get_planning_scene_monitor().read_only() as scene:
                    rs = scene.current_state
                    rs.set_joint_group_positions(group_name, angles)
                    rs.update()
                    tf  = rs.get_global_link_transform(eef_link)
                    pos = np.array([float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3])])
                    positions.append(pos)
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                log_fn(f"  {i+1}/{n_samples} サンプル完了")

        log_fn(f"[サンプリング完了] {len(positions)} 点取得")
        return positions

    def publish_workspace_markers(self, positions: list[np.ndarray], color_rgb=(0.2, 0.8, 1.0)):
        """到達可能点群を /workspace_markers に配信。"""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 既存マーカをクリア
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        r, g, b = color_rgb
        for i, pos in enumerate(positions):
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp    = now
            m.ns     = "workspace"
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.012
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.6
            m.lifetime.sec = 0
            arr.markers.append(m)

        self._pub.publish(arr)

    def publish_trajectory_line(self, positions: "list[np.ndarray]", color_rgb=(0.3, 1.0, 0.3)):
        """スイープで得た EEF 軌跡を LINE_STRIP で /workspace_markers に配信。"""
        if len(positions) < 2:
            return
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        r, g, b = color_rgb
        line = Marker()
        line.header.frame_id = "base_link"
        line.header.stamp    = now
        line.ns              = "j7_sweep_line"
        line.id              = 9000
        line.type            = Marker.LINE_STRIP
        line.action          = Marker.ADD
        line.scale.x         = 0.007
        line.color.r = r; line.color.g = g; line.color.b = b; line.color.a = 1.0
        line.lifetime.sec    = 0
        for pos in positions:
            line.points.append(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
        arr.markers.append(line)
        # 始点・終点を目立たせる
        for sphere_id, pos, alpha in [(9001, positions[0], 0.8), (9002, positions[-1], 1.0)]:
            s = Marker()
            s.header.frame_id = "base_link"
            s.header.stamp    = now
            s.ns              = "j7_sweep_ends"
            s.id              = sphere_id
            s.type            = Marker.SPHERE
            s.action          = Marker.ADD
            s.pose.position.x = float(pos[0])
            s.pose.position.y = float(pos[1])
            s.pose.position.z = float(pos[2])
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.025
            s.color.r = r; s.color.g = g; s.color.b = b; s.color.a = alpha
            s.lifetime.sec = 0
            arr.markers.append(s)
        self._pub.publish(arr)

    def clear_markers(self):
        arr = MarkerArray()
        m = Marker(); m.action = Marker.DELETEALL
        arr.markers.append(m)
        self._pub.publish(arr)


# ── GUI アプリ ─────────────────────────────────────────────────────────────────
class RangeCheckRvizApp:
    def __init__(self):
        rclpy.init()
        self._node = WorkspaceNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        self._result_rows: list[dict] = []
        self._bg_thread: "threading.Thread | None" = None

        self._build_gui()

    def _busy(self) -> bool:
        return self._bg_thread is not None and self._bg_thread.is_alive()

    def _run_bg(self, fn, *args):
        if self._busy():
            self._log("[警告] 別の処理が実行中です")
            return
        self._bg_thread = threading.Thread(target=fn, args=args, daemon=True)
        self._bg_thread.start()

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("sciurus17 可動域調査 — 仮想空間版")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.geometry("780x700")

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(top, bg=PANEL_BG, width=420)
        right = tk.Frame(top, bg=PANEL_BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=4)
        left.pack_propagate(False)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 4), pady=4)

        self._build_controls(left)
        self._build_log(right)

    def _build_controls(self, parent):
        def btn(text, cmd, color=BTN_BG):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg=BTN_FG, font=("Helvetica", 9),
                          relief=tk.FLAT, pady=4)
            b.pack(fill=tk.X, padx=6, pady=2)
            return b

        def sec(text):
            tk.Label(parent, text=text, bg=PANEL_BG, fg="#aaaaaa",
                     font=("Helvetica", 8, "bold")).pack(anchor=tk.W, padx=6, pady=(10, 0))

        def row_lbl_widget(label, widget_fn):
            f = tk.Frame(parent, bg=PANEL_BG)
            f.pack(fill=tk.X, padx=6, pady=1)
            tk.Label(f, text=label, bg=PANEL_BG, fg="#cccccc",
                     font=("Helvetica", 8), width=12, anchor=tk.W).pack(side=tk.LEFT)
            return widget_fn(f)

        # グループ選択
        sec("── グループ ──")
        self._group_var = tk.StringVar(value=list(ARM_GROUPS.keys())[0])
        grp_om = tk.OptionMenu(parent, self._group_var, *ARM_GROUPS.keys())
        grp_om.config(bg=BTN_BG, fg=BTN_FG, font=("Helvetica", 9), relief=tk.FLAT)
        grp_om.pack(fill=tk.X, padx=6, pady=2)

        # 関節制限
        sec("── 関節制限 (URDF) ──")
        btn("関節制限を表示", self._on_show_limits, "#2a4a6a")

        # 関節スイープ
        sec("── 関節スイープ ──")
        self._sweep_joint_var = tk.IntVar(value=0)
        row_lbl_widget("関節番号 (0-6):",
            lambda f: tk.Spinbox(f, from_=0, to=6, textvariable=self._sweep_joint_var,
                                 width=5, bg="#444444", fg="white",
                                 font=("Courier", 9)).pack(side=tk.LEFT))
        self._sweep_steps_var = tk.IntVar(value=10)
        row_lbl_widget("ステップ数:",
            lambda f: tk.Spinbox(f, from_=2, to=50, textvariable=self._sweep_steps_var,
                                 width=5, bg="#444444", fg="white",
                                 font=("Courier", 9)).pack(side=tk.LEFT))
        self._sweep_vel_var = tk.DoubleVar(value=0.3)
        row_lbl_widget("速度スケール:",
            lambda f: tk.Spinbox(f, from_=0.05, to=1.0, increment=0.05,
                                 textvariable=self._sweep_vel_var,
                                 width=5, bg="#444444", fg="white",
                                 font=("Courier", 9)).pack(side=tk.LEFT))
        btn("関節スイープ 開始", self._on_sweep_start, "#2a5a3a")
        btn("スイープ 中断",     self._on_stop,        "#4a2a2a")

        # ランダムサンプリング
        sec("── 到達可能空間サンプリング ──")
        self._n_samples_var = tk.IntVar(value=500)
        row_lbl_widget("サンプル数:",
            lambda f: tk.Spinbox(f, from_=50, to=5000, increment=50,
                                 textvariable=self._n_samples_var,
                                 width=6, bg="#444444", fg="white",
                                 font=("Courier", 9)).pack(side=tk.LEFT))
        btn("ランダムサンプリング 開始", self._on_sample_start, "#2a3a5a")
        btn("サンプリング 中断",        self._on_stop,         "#4a2a2a")
        btn("RViz マーカをクリア",      self._on_clear_markers,"#3a2a4a")

        # 保存
        sec("── CSV 保存 ──")
        self._save_var = tk.StringVar(value="/sciurus17_m2/range_rviz.csv")
        tk.Entry(parent, textvariable=self._save_var,
                 bg="#444444", fg="white", font=("Courier", 8)
                 ).pack(fill=tk.X, padx=6, pady=2)
        btn("CSV 保存", self._on_save_csv, "#2a3a5a")

    def _build_log(self, parent):
        tk.Label(parent, text="ログ", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=6, pady=(6, 0))
        self._log_text = scrolledtext.ScrolledText(
            parent, bg=LOG_BG, fg=LOG_FG, font=("Courier", 8),
            state=tk.DISABLED,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── コールバック ─────────────────────────────────────────────────────────────
    def _on_show_limits(self):
        group = ARM_GROUPS[self._group_var.get()]
        def run():
            try:
                limits = self._node.get_joint_limits(group)
                self._log(f"[関節制限] {group}")
                self._log(f"  {'関節名':<22} {'最小[deg]':>10}  {'最大[deg]':>10}  {'範囲[deg]':>10}")
                for jn, lo, hi in limits:
                    lo_d = math.degrees(lo)
                    hi_d = math.degrees(hi)
                    self._log(f"  {jn:<22} {lo_d:>10.1f}  {hi_d:>10.1f}  {(hi_d-lo_d):>10.1f}")
            except Exception as e:
                self._log(f"[エラー] {e}")
        self._run_bg(run)

    def _on_sweep_start(self):
        group  = ARM_GROUPS[self._group_var.get()]
        j_idx  = self._sweep_joint_var.get()
        steps  = self._sweep_steps_var.get()
        vel    = self._sweep_vel_var.get()

        def run():
            try:
                positions = self._node.sweep_joint(group, j_idx, steps, vel, self._log)
                if positions:
                    self._node.publish_workspace_markers(positions, color_rgb=(0.3, 1.0, 0.3))
                    self._node.publish_trajectory_line(positions, color_rgb=(1.0, 0.9, 0.2))
                    self._log(f"[スイープ完了] {len(positions)} 点 → /workspace_markers 配信 (軌跡LINE_STRIP付き)")
                    for pos in positions:
                        self._result_rows.append({
                            "type": "sweep", "group": group, "joint_index": j_idx,
                            "eef_x": f"{pos[0]:.4f}", "eef_y": f"{pos[1]:.4f}", "eef_z": f"{pos[2]:.4f}",
                        })
            except Exception as e:
                self._log(f"[エラー] {e}")
        self._run_bg(run)

    def _on_sample_start(self):
        group     = ARM_GROUPS[self._group_var.get()]
        n_samples = self._n_samples_var.get()

        def run():
            try:
                positions = self._node.sample_workspace(group, n_samples, self._log)
                if positions:
                    self._node.publish_workspace_markers(positions, color_rgb=(0.2, 0.7, 1.0))
                    self._log(f"[サンプリング完了] {len(positions)} 点 → /workspace_markers 配信")
                    for pos in positions:
                        self._result_rows.append({
                            "type": "sample", "group": group, "joint_index": -1,
                            "eef_x": f"{pos[0]:.4f}", "eef_y": f"{pos[1]:.4f}", "eef_z": f"{pos[2]:.4f}",
                        })
            except Exception as e:
                self._log(f"[エラー] {e}")
        self._run_bg(run)

    def _on_stop(self):
        self._node._stop_flag = True
        self._log("[中断] 停止フラグを設定しました")

    def _on_clear_markers(self):
        self._node.clear_markers()
        self._log("[マーカクリア] /workspace_markers をクリアしました")

    def _on_save_csv(self):
        if not self._result_rows:
            self._log("[CSV] 結果データがありません")
            return
        path = self._save_var.get().strip()
        try:
            fields = list(self._result_rows[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(self._result_rows)
            self._log(f"[CSV保存完了] {len(self._result_rows)} 行 → {path}")
        except Exception as e:
            self._log(f"[CSV エラー] {e}")

    def _log(self, msg: str):
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        def _up():
            self._log_text.config(state=tk.NORMAL)
            self._log_text.insert(tk.END, line)
            self._log_text.see(tk.END)
            self._log_text.config(state=tk.DISABLED)
        self.root.after(0, _up)

    def _on_close(self):
        self._node._stop_flag = True
        rclpy.shutdown()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    RangeCheckRvizApp().run()
