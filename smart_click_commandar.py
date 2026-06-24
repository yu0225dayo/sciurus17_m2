#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import threading
import tkinter as tk
from tkinter import ttk
import math
import sys

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, JointConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, PointStamped

# ==========================================
# グループ名と関節名の定義
# ==========================================
ROBOT_CONFIG = {
    'waist': {
        'group': 'waist_group',
        'joints': ['waist_yaw_joint']
    },
    'l_arm': {
        'group': 'l_arm_waist_group',  # l_arm_group → l_arm_waist_group
        'type': 'arm', 'ee_link': 'l_link7',
        'joints': ['waist_yaw_joint', 'l_arm_joint1', 'l_arm_joint2', 'l_arm_joint3',    'l_arm_joint4', 'l_arm_joint5', 'l_arm_joint6', 'l_arm_joint7']
    },
    'r_arm': {
        'group': 'r_arm_waist_group',  # r_arm_group → r_arm_waist_group
        'type': 'arm', 'ee_link': 'r_link7',
        'joints': ['waist_yaw_joint', 'r_arm_joint1', 'r_arm_joint2', 'r_arm_joint3', 'r_arm_joint4', 'r_arm_joint5', 'r_arm_joint6', 'r_arm_joint7']
    },
    'l_gripper': {
        'group': 'l_gripper_group', 'type': 'gripper',
        'joints': ['l_gripper_joint']
    },
    'r_gripper': {
        'group': 'r_gripper_group', 'type': 'gripper',
        'joints': ['r_gripper_joint']
    },
    'neck': {
        'group': 'neck_group', 'type': 'neck',
        'joints': ['neck_yaw_joint', 'neck_pitch_joint']
    },
}

# ==========================================
# 初期化時の各関節の目標角度（ラジアン）
# 実機の/joint_statesデータに基づく安全な待機姿勢
# ==========================================
INIT_POSES = {
    'waist': [0.0046019423636569235],
    'l_arm': [
        0.0046019423636569235,     # waist_yaw_joint ← 追加
        0.007669903939428206,      # l_arm_joint1
        1.5677283652191252,        # l_arm_joint2
        -0.0015339807878856412,    # l_arm_joint3
        -2.724349879284899,        # l_arm_joint4
        -0.0030679615757712823,    # l_arm_joint5
        2.0923497946760143,        # l_arm_joint6
        0.0                        # l_arm_joint7
    ],
    'r_arm': [
        0.0046019423636569235,     # waist_yaw_joint ← 追加
        -0.0046019423636569235,    # r_arm_joint1
        -1.569262346007011,        # r_arm_joint2
        0.0,                       # r_arm_joint3
        2.6952042443150717,        # r_arm_joint4
        0.0,                       # r_arm_joint5
        -2.086213871524472,        # r_arm_joint6
        0.0                        # r_arm_joint7
    ],
    'l_gripper': [-0.7148350471547088],
    'r_gripper': [0.8605632220038447],
    'neck': [
        0.006135923151542565,      # neck_yaw_joint
        -0.6580777580029401        # neck_pitch_joint
    ]
}

def quaternion_from_euler(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]


class SmartClickCommander(Node):
    def __init__(self):
        super().__init__('smart_click_commander')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        
        self.point_sub = self.create_subscription(PointStamped, '/clicked_point', self.clicked_point_callback, 10)
        
        self.ui_callback = None
        self.latest_goal_req = None
        self.latest_target_group = "l_arm"
        
        # 💡 微調整用に現在の目標座標と姿勢を保持する変数
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_q = None
        
        self.init_queue = ['waist', 'l_arm', 'r_arm', 'l_gripper', 'r_gripper', 'neck']
        
        self.init_timer = self.create_timer(2.0, self._start_init_sequence)
        self.get_logger().info("System initialized. Waiting for ROS 2 network stabilization...")

    def _start_init_sequence(self):
        self.init_timer.cancel()
        if self.ui_callback:
            self.ui_callback("⚙️ 全身の初期化シーケンスを実行中...\nロボットから離れてください。")
        self.get_logger().info("--- INITIALIZATION SEQUENCE STARTED ---")
        self._process_next_init_task()

    def _process_next_init_task(self):
        if not self.init_queue:
            self.get_logger().info("--- ALL INITIALIZATION TASKS COMPLETED ---")
            if self.ui_callback:
                self.ui_callback("✅ 初期化完了。\nRVizで対象をクリックしてPlanを開始してください。\n\n【終了は 'q' キー】")
            return

        target_part = self.init_queue.pop(0)
        config = ROBOT_CONFIG.get(target_part)
        target_angles = INIT_POSES.get(target_part)

        if not config or target_angles is None:
            self.get_logger().warn(f"Task {target_part} missing config. Skipping...")
            self._process_next_init_task()
            return

        self.get_logger().info(f"Initializing [{target_part}]...")
        self._send_joint_goal_with_callback(
            config['group'], 
            config['joints'], 
            target_angles, 
            on_done=self._on_init_task_completed
        )

    def _on_init_task_completed(self, success):
        if not success:
            self.get_logger().error("⚠️ 初期化タスクの1つが失敗しました。シーケンスを中断します。")
            if self.ui_callback:
                self.ui_callback("❌ 初期化エラー。ターミナルを確認してください。")
            return
        self._process_next_init_task()

    def _send_joint_goal_with_callback(self, group_name, joint_names, target_angles, on_done=None):
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("MoveGroup Action Server not available.")
            if on_done: on_done(False)
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = group_name
        req.allowed_planning_time = 3.0
        req.start_state.is_diff = True

        constraints = Constraints()

        for j_name, t_angle in zip(joint_names, target_angles):
            jc = JointConstraint()
            jc.joint_name = j_name
            jc.position = float(t_angle)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)
        goal_msg.request = req
        goal_msg.planning_options.plan_only = False

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        if on_done:
            send_goal_future.add_done_callback(lambda fut: self._goal_response_callback(fut, on_done))

    def _goal_response_callback(self, future, on_done):
        goal_handle = future.result()
        if not goal_handle.accepted:
            if on_done: on_done(False)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda res_future: self._goal_result_callback(res_future, on_done)
        )

    def _goal_result_callback(self, future, on_done):
        status = future.result().status
        success = (status == 4) 
        if on_done: on_done(success)

    def execute_gripper_macro(self, config_key, target_angle):
        config = ROBOT_CONFIG[config_key]
        self._send_joint_goal_with_callback(config['group'], config['joints'], [target_angle])

    def execute_neck_macro(self, tilt_deg, pan_deg):
        config = ROBOT_CONFIG['neck']
        angles = [math.radians(pan_deg), math.radians(tilt_deg)]
        self._send_joint_goal_with_callback(config['group'], config['joints'], angles)

    def clicked_point_callback(self, msg: PointStamped):
        cx, cy, cz = msg.point.x, msg.point.y, msg.point.z
        dist = math.sqrt(cx**2 + cy**2)
        if dist < 0.1:
            self.get_logger().warn("クリック位置が近すぎます。")
            return
            
        offset = 0.04
        
        # 💡 ここで計算した目標座標をクラスの変数として記録しておく
        self.target_x = cx - offset * (cx / dist)
        self.target_y = cy 
        self.target_z = cz + 0.05
        
        FIXED_YAW_DEG = -90.0
        target_roll_deg = 0.0
        target_pitch_deg = 0.0
        yaw = math.radians(FIXED_YAW_DEG)
        target_roll = math.radians(target_roll_deg)
        target_pitch = math.radians(target_pitch_deg)
        
        # 💡 姿勢も記録
        self.target_q = quaternion_from_euler(target_roll, target_pitch, yaw)

        if self.ui_callback:
            self.ui_callback(f"Target Set: X={self.target_x:.2f}, Y={self.target_y:.2f}, Z={self.target_z:.2f}\n(微調整ボタンで移動可能)")

        config = ROBOT_CONFIG[self.latest_target_group]
        self._generate_and_send_arm_plan(config, self.target_x, self.target_y, self.target_z, self.target_q)

    # 💡 新機能：目標座標を相対的に微調整して再度プランニングを要求する
    def nudge_target(self, dx, dy, dz):
        if self.target_x is None:
            self.get_logger().warn("RVizで対象をクリックしていません。")
            if self.ui_callback:
                self.ui_callback("⚠️ 先にRViz上で対象をクリックしてください。")
            return

        # 座標を増減
        self.target_x += dx
        self.target_y += dy
        self.target_z += dz

        if self.ui_callback:
            self.ui_callback(f"Target Nudged: X={self.target_x:.2f}, Y={self.target_y:.2f}, Z={self.target_z:.2f}\n(再プランニング完了)")

        config = ROBOT_CONFIG[self.latest_target_group]
        self._generate_and_send_arm_plan(config, self.target_x, self.target_y, self.target_z, self.target_q)

    def _generate_and_send_arm_plan(self, config, x, y, z, q, execute=False):
        if not self._action_client.wait_for_server(timeout_sec=1.0): return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = config['group']
        req.allowed_planning_time = 2.0
        req.start_state.is_diff = True
        
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name = config['ee_link']
        primitive = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])
        pc.constraint_region.primitives.append(primitive)
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = config['ee_link']
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = q
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0

        req.goal_constraints.append(Constraints(position_constraints=[pc], orientation_constraints=[oc]))
        goal_msg.request = req
        goal_msg.planning_options.plan_only = not execute

        self.latest_goal_req = goal_msg
        self._action_client.send_goal_async(goal_msg)

    def execute_saved_plan(self):
        if self.latest_goal_req:
            self.latest_goal_req.planning_options.plan_only = False
            self._action_client.send_goal_async(self.latest_goal_req)
            self.latest_goal_req = None
            if self.ui_callback:
                self.ui_callback("実機へ実行コマンドを送信しました！\nRViz上で次のポイントをクリックしてください。")


# ========== Tkinter UI アプリケーション ==========
class IntegratedUI:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.node.ui_callback = self.update_status
        self.root.title("Smart Click Commander")
        # 💡 UIの縦幅を少し広げる
        self.root.geometry("380x600")

        self.root.bind('<q>', self.quit_app)
        self.root.bind('<Q>', self.quit_app)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.target_arm = tk.StringVar(value="l_arm")
        self._create_widgets()

    def _create_widgets(self):
        frame_arm = tk.LabelFrame(self.root, text="1. 操作する腕を選択", padx=10, pady=5)
        frame_arm.pack(fill="x", padx=10, pady=5)
        ttk.Radiobutton(frame_arm, text="Left Arm", variable=self.target_arm, value="l_arm", command=self.on_arm_change).pack(side="left", padx=10)
        ttk.Radiobutton(frame_arm, text="Right Arm", variable=self.target_arm, value="r_arm", command=self.on_arm_change).pack(side="right", padx=10)

        self.status_label = tk.Label(self.root, text="システム起動中...\n(初期化シーケンスの完了をお待ちください)", 
                                     bg="#e1f5fe", height=5, relief="sunken")
        self.status_label.pack(fill="x", padx=10, pady=10)

        # 💡 追加：5. 目標位置の微調整 (X/Y/Z)
        frame_jog = tk.LabelFrame(self.root, text="2. ゴーストの微調整 (±1cm)", padx=10, pady=5)
        frame_jog.pack(fill="x", padx=10, pady=5)
        
        # グリッドレイアウトで直感的に配置
        tk.Button(frame_jog, text="X +1cm (奥)", bg="#f5f5f5", command=lambda: self.node.nudge_target(0.01, 0, 0)).grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        tk.Button(frame_jog, text="X -1cm (手前)", bg="#f5f5f5", command=lambda: self.node.nudge_target(-0.01, 0, 0)).grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        tk.Button(frame_jog, text="Y +1cm (左)", bg="#f5f5f5", command=lambda: self.node.nudge_target(0, 0.01, 0)).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        tk.Button(frame_jog, text="Y -1cm (右)", bg="#f5f5f5", command=lambda: self.node.nudge_target(0, -0.01, 0)).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        tk.Button(frame_jog, text="Z +1cm (上)", bg="#f5f5f5", command=lambda: self.node.nudge_target(0, 0, 0.01)).grid(row=0, column=2, padx=5, pady=2, sticky="ew")
        tk.Button(frame_jog, text="Z -1cm (下)", bg="#f5f5f5", command=lambda: self.node.nudge_target(0, 0, -0.01)).grid(row=1, column=2, padx=5, pady=2, sticky="ew")
        
        # 列幅を均等にする
        frame_jog.grid_columnconfigure(0, weight=1)
        frame_jog.grid_columnconfigure(1, weight=1)
        frame_jog.grid_columnconfigure(2, weight=1)

        frame_exec = tk.LabelFrame(self.root, text="3. ゴースト確認後に実行", padx=10, pady=10)
        frame_exec.pack(fill="x", padx=10, pady=5)
        btn_exec = tk.Button(frame_exec, text="🚀 ゴーストの位置へ実機を移動", bg="#ffcdd2", fg="black", font=("Arial", 11, "bold"),
                             command=self.node.execute_saved_plan, height=2)
        btn_exec.pack(fill="x")

        frame_grip = tk.LabelFrame(self.root, text="4. グリッパー操作 (即時実行)", padx=10, pady=5)
        frame_grip.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_grip, text="L_Gripper (0°)", bg="#e0f7fa", command=lambda: self.node.execute_gripper_macro('l_gripper', 0.0)).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(frame_grip, text="R_Gripper (0°)", bg="#fff3e0", command=lambda: self.node.execute_gripper_macro('r_gripper', 0.0)).pack(side="right", expand=True, fill="x", padx=2)

        frame_neck = tk.LabelFrame(self.root, text="5. カメラアングル", padx=10, pady=5)
        frame_neck.pack(fill="x", padx=10, pady=5)
        tk.Button(frame_neck, text="👀 首（Neck）を下向き固定", bg="#f3e5f5", 
                  command=lambda: self.node.execute_neck_macro(-60.0, 0.0)).pack(fill="x", pady=2)

    def on_arm_change(self):
        self.node.latest_target_group = self.target_arm.get()
        self.update_status(f"ターゲットを {self.target_arm.get()} に変更しました。\nRVizで対象をクリックしてください。\n\n【終了は 'q' キーを押す】")

    def update_status(self, text):
        self.status_label.config(text=text)

    def quit_app(self, event=None):
        self.node.get_logger().info("Quit key ('q') pressed. Shutting down system...")
        self.root.quit()
        self.root.destroy()

def spin_ros(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    node = SmartClickCommander()
    
    spin_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    spin_thread.start()
    
    root = tk.Tk()
    app = IntegratedUI(root, node)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)

if __name__ == '__main__':
    main()
