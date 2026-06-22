#!/usr/bin/env python3
"""
sciurus17 可動域調査 — 実機版

起動手順:
  1. hardware.launch.py でロボットを起動
  2. 本スクリプトを実行:
       python3 /sciurus17_m2/ros/sciurus17_range_realbot.py
  3. [コントローラ停止] でアームトルクを無効化
  4. 人間がアームを手動で動かす
  5. [現在姿勢を記録] または [連続記録] でログ蓄積
  6. [CSV保存] で /sciurus17_m2/range_log.csv に書き出し

依存:
  ROS2 Jazzy + hardware.launch.py 起動済み
  pip: numpy
"""

import csv
import datetime
import math
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import numpy as np

try:
    import rclpy
    import rclpy.duration
    import rclpy.time
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError as e:
    print(f"[エラー] ROS2 が見つかりません: {e}")
    sys.exit(1)

_TF2_OK = False
try:
    from tf2_ros import Buffer, TransformListener
    _TF2_OK = True
except ImportError:
    pass

# ── 定数 ───────────────────────────────────────────────────────────────────────
MONITOR_JOINTS = [
    "l_arm_joint1", "l_arm_joint2", "l_arm_joint3",
    "l_arm_joint4", "l_arm_joint5", "l_arm_joint6", "l_arm_joint7",
    "r_arm_joint1", "r_arm_joint2", "r_arm_joint3",
    "r_arm_joint4", "r_arm_joint5", "r_arm_joint6", "r_arm_joint7",
    "waist_yaw_joint",
]
EEF_LINKS  = ["l_link7", "r_link7"]
BASE_FRAME = "base_link"

ARM_CONTROLLERS   = ["left_arm_controller", "right_arm_controller"]
WAIST_CONTROLLERS = ["waist_yaw_controller"]

BG = "#2b2b2b"; PANEL_BG = "#3c3f41"; BTN_BG = "#555759"; BTN_FG = "white"
LOG_BG = "#1e1e1e"; LOG_FG = "#cccccc"


# ── ROS2 ノード ────────────────────────────────────────────────────────────────
class MonitorNode(Node):
    def __init__(self):
        super().__init__("sciurus17_range_realbot")
        self._joint_pos: dict[str, float] = {}
        self._sub = self.create_subscription(
            JointState, "/joint_states", self._cb_joint, 10
        )
        self._tf_buf = None
        if _TF2_OK:
            self._tf_buf = Buffer()
            self._tf_listener = TransformListener(self._tf_buf, self)

    def _cb_joint(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_pos[name] = pos

    def get_eef_positions(self) -> dict[str, "np.ndarray | None"]:
        if self._tf_buf is None:
            return {l: None for l in EEF_LINKS}
        result = {}
        for link in EEF_LINKS:
            try:
                t = self._tf_buf.lookup_transform(
                    BASE_FRAME, link, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
                tr = t.transform.translation
                result[link] = np.array([tr.x, tr.y, tr.z])
            except Exception:
                result[link] = None
        return result


# ── GUI アプリ ─────────────────────────────────────────────────────────────────
class RangeCheckRealApp:
    def __init__(self):
        rclpy.init()
        self._node = MonitorNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        self._log_rows: list[dict] = []
        self._joint_min: dict[str, float] = {}
        self._joint_max: dict[str, float] = {}
        self._continuous = False

        self._build_gui()

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("sciurus17 可動域調査 — 実機版")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.geometry("820x680")

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(top, bg=PANEL_BG)
        right = tk.Frame(top, bg=PANEL_BG, width=260)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 2), pady=4)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=4)
        right.pack_propagate(False)

        self._build_joint_table(left)
        self._build_eef_panel(left)
        self._build_controls(right)

        self.root.after(150, self._update_loop)

    def _build_joint_table(self, parent):
        tk.Label(parent, text="関節角度 (リアルタイム)",
                 bg=PANEL_BG, fg="#aaaaaa", font=("Helvetica", 9, "bold")
                 ).pack(anchor=tk.W, padx=6, pady=(6, 0))

        frm = tk.Frame(parent, bg=PANEL_BG)
        frm.pack(fill=tk.X, padx=6)

        hdrs = ["関節名", "現在値[deg]", "最小[deg]", "最大[deg]"]
        widths = [18, 11, 11, 11]
        fgs    = ["#aaaaaa", "#88ff88", "#ffaa44", "#44aaff"]
        for c, (h, w, fg) in enumerate(zip(hdrs, widths, fgs)):
            tk.Label(frm, text=h, bg=PANEL_BG, fg=fg,
                     font=("Courier", 8), width=w, anchor=tk.W
                     ).grid(row=0, column=c, sticky=tk.W)

        self._jnt_vars: dict[str, tuple] = {}
        for i, jname in enumerate(MONITOR_JOINTS):
            row = i + 1
            cur = tk.StringVar(value="--")
            mn  = tk.StringVar(value="--")
            mx  = tk.StringVar(value="--")
            tk.Label(frm, text=jname, bg=PANEL_BG, fg="white",
                     font=("Courier", 8), width=18, anchor=tk.W
                     ).grid(row=row, column=0, sticky=tk.W)
            tk.Label(frm, textvariable=cur, bg=PANEL_BG, fg="#88ff88",
                     font=("Courier", 8), width=11).grid(row=row, column=1)
            tk.Label(frm, textvariable=mn, bg=PANEL_BG, fg="#ffaa44",
                     font=("Courier", 8), width=11).grid(row=row, column=2)
            tk.Label(frm, textvariable=mx, bg=PANEL_BG, fg="#44aaff",
                     font=("Courier", 8), width=11).grid(row=row, column=3)
            self._jnt_vars[jname] = (cur, mn, mx)

    def _build_eef_panel(self, parent):
        tk.Label(parent, text="EEF 位置 (base_link 座標 [m])",
                 bg=PANEL_BG, fg="#aaaaaa", font=("Helvetica", 9, "bold")
                 ).pack(anchor=tk.W, padx=6, pady=(8, 0))
        self._eef_l_var = tk.StringVar(value="L l_link7: --")
        self._eef_r_var = tk.StringVar(value="R r_link7: --")
        for var, fg in [(self._eef_l_var, "#88ff88"), (self._eef_r_var, "#ff8888")]:
            tk.Label(parent, textvariable=var, bg=PANEL_BG, fg=fg,
                     font=("Courier", 9)).pack(anchor=tk.W, padx=10)

    def _build_controls(self, parent):
        def btn(text, cmd, color=BTN_BG, pady=4):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=color, fg=BTN_FG, font=("Helvetica", 9),
                          relief=tk.FLAT, pady=pady)
            b.pack(fill=tk.X, padx=6, pady=2)
            return b

        def sec(text):
            tk.Label(parent, text=text, bg=PANEL_BG, fg="#aaaaaa",
                     font=("Helvetica", 8, "bold")).pack(anchor=tk.W, padx=6, pady=(10, 0))

        sec("── コントローラ制御 ──")
        btn("コントローラ停止 (トルク無効)", self._on_deactivate, "#5a2020")
        btn("コントローラ起動 (トルク有効)", self._on_activate,   "#1a4a2a")

        sec("── 記録 ──")
        btn("現在姿勢を記録", self._on_record_once, "#2a4a5a")
        self._cont_btn = btn("連続記録 開始", self._on_toggle_continuous, "#3a3a1a")
        btn("記録リセット",   self._on_reset_records, "#4a2a2a")

        sec("── CSV 保存 ──")
        self._save_var = tk.StringVar(value="/sciurus17_m2/range_log.csv")
        tk.Entry(parent, textvariable=self._save_var,
                 bg="#444444", fg="white", font=("Courier", 8)
                 ).pack(fill=tk.X, padx=6, pady=2)
        btn("CSV 保存", self._on_save_csv, "#2a3a5a")

        sec("── ログ ──")
        self._log_text = scrolledtext.ScrolledText(
            parent, bg=LOG_BG, fg=LOG_FG, font=("Courier", 8),
            state=tk.DISABLED,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── 定期更新 ────────────────────────────────────────────────────────────────
    def _update_loop(self):
        jp  = dict(self._node._joint_pos)
        eef = self._node.get_eef_positions()

        for jname, (cur_v, mn_v, mx_v) in self._jnt_vars.items():
            pos = jp.get(jname)
            if pos is None:
                continue
            deg = math.degrees(pos)
            cur_v.set(f"{deg:+7.2f}")
            if jname not in self._joint_min or deg < self._joint_min[jname]:
                self._joint_min[jname] = deg
                mn_v.set(f"{deg:+7.2f}")
            if jname not in self._joint_max or deg > self._joint_max[jname]:
                self._joint_max[jname] = deg
                mx_v.set(f"{deg:+7.2f}")

        lp = eef.get("l_link7")
        rp = eef.get("r_link7")
        self._eef_l_var.set("L l_link7: " + (
            f"X={lp[0]:+.3f}  Y={lp[1]:+.3f}  Z={lp[2]:+.3f}" if lp is not None else "--"))
        self._eef_r_var.set("R r_link7: " + (
            f"X={rp[0]:+.3f}  Y={rp[1]:+.3f}  Z={rp[2]:+.3f}" if rp is not None else "--"))

        if self._continuous:
            self._do_record(jp, eef)

        self.root.after(150, self._update_loop)

    # ── コールバック ─────────────────────────────────────────────────────────────
    def _on_deactivate(self):
        self._run_ros2_control(
            ["--deactivate"] + ARM_CONTROLLERS + WAIST_CONTROLLERS,
            "コントローラ停止 → アームのトルクが無効になりました。手動で動かせます。",
        )

    def _on_activate(self):
        self._run_ros2_control(
            ["--activate"] + ARM_CONTROLLERS + WAIST_CONTROLLERS,
            "コントローラ起動 → トルクが有効になりました。",
        )

    def _run_ros2_control(self, extra_args: list[str], msg: str):
        cmd = ["ros2", "control", "switch_controllers"] + extra_args
        def run():
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    self._log(msg)
                else:
                    self._log(f"[エラー] {r.stderr.strip() or r.stdout.strip()}")
            except Exception as e:
                self._log(f"[エラー] {e}")
        threading.Thread(target=run, daemon=True).start()

    def _on_record_once(self):
        jp  = dict(self._node._joint_pos)
        eef = self._node.get_eef_positions()
        if not jp:
            self._log("[記録] /joint_states 未受信 — ロボットが起動しているか確認してください")
            return
        self._do_record(jp, eef)
        self._log(f"[記録] {len(self._log_rows)} 件目を記録しました")

    def _on_toggle_continuous(self):
        self._continuous = not self._continuous
        self._cont_btn.config(
            text="連続記録 停止" if self._continuous else "連続記録 開始",
            bg="#5a5a00"     if self._continuous else "#3a3a1a",
        )
        self._log("連続記録 " + ("開始" if self._continuous else "停止"))

    def _on_reset_records(self):
        self._log_rows.clear()
        self._joint_min.clear()
        self._joint_max.clear()
        for _, (_, mn_v, mx_v) in self._jnt_vars.items():
            mn_v.set("--"); mx_v.set("--")
        self._log("[リセット] 記録・可動域データをクリアしました")

    def _do_record(self, jp: dict, eef: dict):
        ts  = datetime.datetime.now().isoformat(timespec="milliseconds")
        row: dict = {"timestamp": ts}
        for jn in MONITOR_JOINTS:
            pos = jp.get(jn)
            row[f"{jn}_rad"] = f"{pos:.6f}"  if pos is not None else ""
            row[f"{jn}_deg"] = f"{math.degrees(pos):.3f}" if pos is not None else ""
        for link in EEF_LINKS:
            p = eef.get(link)
            if p is not None:
                row[f"{link}_x"] = f"{p[0]:.4f}"
                row[f"{link}_y"] = f"{p[1]:.4f}"
                row[f"{link}_z"] = f"{p[2]:.4f}"
            else:
                row[f"{link}_x"] = row[f"{link}_y"] = row[f"{link}_z"] = ""
        self._log_rows.append(row)

    def _on_save_csv(self):
        if not self._log_rows:
            self._log("[CSV] 記録データがありません")
            return
        path = self._save_var.get().strip()
        try:
            fields = list(self._log_rows[0].keys())
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
                csv.DictWriter(f, fieldnames=fields).writerows(self._log_rows)
            self._log(f"[CSV保存完了] {len(self._log_rows)} 行 → {path}")
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
        self._continuous = False
        rclpy.shutdown()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    RangeCheckRealApp().run()
