#!/usr/bin/env python3
"""
sciurus17 把持パイプライン — ROS2 ノード

フロー:
  Step 0: sciurus17 の RealSense カメラトピック (RGBD) を取得
  Step 1: HTTP で計算機 (server.py) に RGBD 送信 → SAM-3D でメッシュ生成
  Step 2: HTTP で計算機に送信 → SAM-6D で 6DoF pose 推定 (R, t)
  Step 3: Shape2Gesture で両手把持姿勢生成 → hand[0] (手首、物体正規化座標)
  Step 4: TF2 変換 camera_color_optical_frame → base_link
  Step 5: MoveItPy で左右アームをプリグラスプ → グラスプ位置へ移動

使い方:
  # ROS2 環境 + sciurus17_ros ビルド済みで
  python3 ros/sciurus17_grasp_pipeline.py \\
      --server-url http://10.40.1.126:8080 \\
      --click-x 320 --click-y 240

  # インタラクティブ選択（ウィンドウでクリック）
  python3 ros/sciurus17_grasp_pipeline.py --server-url http://10.40.1.126:8080

依存:
  ROS2 (rclpy, tf2_ros, tf2_geometry_msgs, sensor_msgs, geometry_msgs, cv_bridge)
  MoveIt2 (moveit)
  pip install requests opencv-python numpy torch
"""

import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PointStamped, PoseStamped, Quaternion
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.logging import get_logger
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs  # noqa: F401 — transform() 用に必要
from tf2_ros import Buffer, TransformListener

# client/ ディレクトリのモジュールを参照
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "client"))

from pipeline.sam6d_detector import SAM6DClient          # noqa: E402
from pipeline.grasp_generator import GraspGenerator      # noqa: E402
from utils.coord_transform import (                       # noqa: E402
    CameraIntrinsics, ObjectPose, normalized_to_camera,
)
from utils.pointcloud_utils import load_pointcloud_ply    # noqa: E402


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _align_y_up(mesh_pts: np.ndarray):
    """PEM は Y 軸下向きで出力するため X 軸周り 180° 回転で補正する。"""
    R_corr = np.diag([1.0, -1.0, -1.0])
    return (R_corr @ mesh_pts.T).T.astype(mesh_pts.dtype), R_corr


def plan_and_execute(robot, planning_component, logger, plan_params) -> bool:
    """軌道を計画して実行する。成功なら True を返す。"""
    plan_result = planning_component.plan(single_plan_parameters=plan_params)
    if plan_result:
        logger.info("軌道計画成功 → 実行中")
        robot.execute(plan_result.trajectory, controllers=[])
        return True
    logger.error("軌道計画失敗")
    return False


def set_gripper(robot, gripper_comp, robot_model, group_name, angle_rad,
                logger, gripper_params):
    """グリッパを指定角度 [rad] に動かす。"""
    rs = RobotState(robot_model)
    rs.set_joint_group_positions(group_name, [angle_rad])
    gripper_comp.set_start_state_to_current_state()
    gripper_comp.set_goal_state(robot_state=rs)
    plan_and_execute(robot, gripper_comp, logger, gripper_params)


# ---------------------------------------------------------------------------
# ROS2 ノード
# ---------------------------------------------------------------------------

class GraspPipelineNode(Node):
    """
    sciurus17 把持パイプライン ノード

    カメラトピックを購読し、フレームが揃ったら把持パイプラインを実行する。
    """

    def __init__(self, args):
        super().__init__("sciurus17_grasp_pipeline")
        self.args = args
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._rgb: np.ndarray | None = None
        self._depth_mm: np.ndarray | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._frame_ready = threading.Event()

        # TF2
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # カメラトピック購読
        self.create_subscription(Image,      args.color_topic, self._color_cb, 1)
        self.create_subscription(Image,      args.depth_topic, self._depth_cb, 1)
        self.create_subscription(CameraInfo, args.info_topic,  self._info_cb,  1)

        self.get_logger().info(
            f"カメラ購読: color={args.color_topic} depth={args.depth_topic}"
        )

    # --- コールバック ---

    def _color_cb(self, msg: Image):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._lock:
                self._rgb = bgr
            self._check_ready()
        except Exception as e:
            self.get_logger().error(f"カラー変換エラー: {e}")

    def _depth_cb(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self._depth_mm = depth.copy()
            self._check_ready()
        except Exception as e:
            self.get_logger().error(f"深度変換エラー: {e}")

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            if self._intrinsics is None:
                self._intrinsics = CameraIntrinsics(
                    fx=msg.k[0], fy=msg.k[4],
                    cx=msg.k[2], cy=msg.k[5],
                    width=msg.width, height=msg.height,
                )
                self.get_logger().info(
                    f"カメラパラメータ取得: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} "
                    f"cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}"
                )
        self._check_ready()

    def _check_ready(self):
        with self._lock:
            ready = (self._rgb is not None and
                     self._depth_mm is not None and
                     self._intrinsics is not None)
        if ready:
            self._frame_ready.set()

    # --- 公開 API ---

    def wait_for_frame(self, timeout: float = 15.0):
        """カラー・深度・内部パラメータが揃うまでブロック。タイムアウト時に例外。"""
        if not self._frame_ready.wait(timeout):
            raise TimeoutError(
                f"カメラフレーム取得タイムアウト ({timeout:.0f}s)。"
                "カメラトピックが配信されているか確認してください。"
            )
        with self._lock:
            return self._rgb.copy(), self._depth_mm.copy(), self._intrinsics

    def camera_to_base(
        self,
        xyz_cam: np.ndarray,
        camera_frame: str,
        base_frame: str,
    ) -> np.ndarray:
        """カメラ座標系 [m] → base_link 座標系 [m] (TF2 変換)"""
        pt = PointStamped()
        pt.header.frame_id = camera_frame
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (
            float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])
        )
        try:
            pt_base = self._tf_buffer.transform(
                pt, base_frame,
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as e:
            raise RuntimeError(
                f"TF2 変換失敗 {camera_frame} → {base_frame}: {e}\n"
                "TF ツリーが構築されているか確認してください。"
            )
        return np.array([pt_base.point.x, pt_base.point.y, pt_base.point.z])

    def move_arm_to(
        self,
        robot,
        arm_comp,
        logger,
        plan_params,
        xyz_base: np.ndarray,
        pose_link: str,
        orientation: Quaternion,
    ) -> bool:
        """base_link 座標系の xyz にアームエンドエフェクタを移動する。"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base_link"
        goal_pose.pose = Pose(
            position=Point(x=float(xyz_base[0]),
                           y=float(xyz_base[1]),
                           z=float(xyz_base[2])),
            orientation=orientation,
        )
        arm_comp.set_start_state_to_current_state()
        arm_comp.set_goal_state(pose_stamped_msg=goal_pose, pose_link=pose_link)
        return plan_and_execute(robot, arm_comp, logger, plan_params)


# ---------------------------------------------------------------------------
# パイプライン本体
# ---------------------------------------------------------------------------

def run_pipeline(node: GraspPipelineNode, args):
    logger = node.get_logger()

    # ------------------------------------------------------------------
    # Step 0: カメラフレーム取得
    # ------------------------------------------------------------------
    logger.info("[Step 0] カメラフレーム待機中...")
    rgb, depth_mm, intrinsics = node.wait_for_frame(timeout=15.0)
    depth_m = depth_mm.astype(np.float32) * args.depth_scale
    logger.info(f"[Step 0完了] shape={rgb.shape} depth_range=[{depth_m.min():.2f}, {depth_m.max():.2f}] m")

    # ------------------------------------------------------------------
    # Step 1 & 2: メッシュ生成 + 6DoF pose 推定
    # ------------------------------------------------------------------
    client = SAM6DClient(
        server_url=args.server_url,
        timeout_mesh=300.0,
        timeout_pose=60.0,
    )

    mesh_path = args.mesh_out
    os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)

    logger.info("[Step 1] SAM-3D でメッシュ生成中...")
    if args.click_x >= 0 and args.click_y >= 0:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            click_x=args.click_x, click_y=args.click_y,
            mesh_method=args.mesh_method,
        )
        click_x, click_y = args.click_x, args.click_y
    else:
        logger.info("[Step 1] インタラクティブ選択: ウィンドウで物体をクリックしてください")
        _, click_x, click_y, masks, scores = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=args.mesh_method,
        )
    logger.info(f"[Step 1完了] mesh 保存: {mesh_path}")

    logger.info("[Step 2] SAM-6D で 6DoF pose 推定中...")
    R, t, _, _ = client.estimate_pose(
        rgb, depth_m, intrinsics, click_x=click_x, click_y=click_y,
    )
    logger.info(
        f"[Step 2完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m\n"
        f"  R=\n{R}"
    )

    # ------------------------------------------------------------------
    # Step 3: Shape2Gesture で両手把持姿勢生成
    # ------------------------------------------------------------------
    logger.info("[Step 3] Shape2Gesture で把持姿勢生成中...")
    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)

    # Y-up 補正 (PEM は Y 軸下向きで出力するため)
    mesh_pts, R_corr = _align_y_up(mesh_pts)
    R = (R.astype(np.float64) @ R_corr.T).astype(np.float32)

    # スケール [m] を点群の最大半径から計算 (mesh_pts は mm 単位)
    _centered = mesh_pts - mesh_pts.mean(axis=0)
    mesh_scale_m = float(np.max(np.linalg.norm(_centered, axis=1))) / 1000.0
    logger.info(f"[Step 3] mesh_scale_m={mesh_scale_m:.4f} m")

    pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)

    generator = GraspGenerator(model_dir=args.model_dir, epoch=args.model_epoch)
    generator.load_models()
    grasp_results = generator.generate(mesh_pts, num_samples=1)
    lh_norm, rh_norm = grasp_results[0]  # (23, 3) 各手の関節座標（正規化系）

    # 正規化座標系 → カメラ座標系
    lh_cam = normalized_to_camera(lh_norm, pose)  # (23, 3) [m]
    rh_cam = normalized_to_camera(rh_norm, pose)

    # hand[0] = 手首 (index 0 は wrist_format=[0,0,0] を起源とする手首位置)
    lh_wrist_cam = lh_cam[0]
    rh_wrist_cam = rh_cam[0]
    logger.info(f"[Step 3完了] 左手首(cam): {lh_wrist_cam}")
    logger.info(f"             右手首(cam): {rh_wrist_cam}")

    # ------------------------------------------------------------------
    # Step 4: TF2 変換 (カメラ座標系 → base_link)
    # ------------------------------------------------------------------
    logger.info(
        f"[Step 4] TF2 変換: {args.camera_frame} → {args.base_frame}"
    )
    lh_base = node.camera_to_base(lh_wrist_cam, args.camera_frame, args.base_frame)
    rh_base = node.camera_to_base(rh_wrist_cam, args.camera_frame, args.base_frame)
    logger.info(f"[Step 4完了] 左手首(base_link): {lh_base}")
    logger.info(f"             右手首(base_link): {rh_base}")

    # ------------------------------------------------------------------
    # Step 5: MoveItPy でアーム移動
    # ------------------------------------------------------------------
    logger.info("[Step 5] MoveItPy でアーム制御開始...")
    robot = MoveItPy(node_name="sciurus17_grasp_pipeline_moveit")

    l_arm     = robot.get_planning_component(args.l_arm_group)
    r_arm     = robot.get_planning_component(args.r_arm_group)
    l_gripper = robot.get_planning_component("l_gripper_group")
    r_gripper = robot.get_planning_component("r_gripper_group")
    robot_model = robot.get_robot_model()

    plan_params = PlanRequestParameters(robot, "ompl_rrtc_default")
    plan_params.max_velocity_scaling_factor = 0.1
    plan_params.max_acceleration_scaling_factor = 0.1

    gripper_params = PlanRequestParameters(robot, "ompl_rrtc_default")
    gripper_params.max_velocity_scaling_factor = 1.0
    gripper_params.max_acceleration_scaling_factor = 1.0

    # グリッパ角度
    GRIPPER_OPEN_L  = math.radians(-40.0)
    GRIPPER_OPEN_R  = math.radians(40.0)
    GRIPPER_GRASP_L = math.radians(-20.0)
    GRIPPER_GRASP_R = math.radians(20.0)

    # 手首の向き: 上方から下向きに把持
    q_down_l = Quaternion(x=-0.707, y=0.0, z=0.0, w=0.707)
    q_down_r = Quaternion(x=0.707,  y=0.0, z=0.0, w=0.707)

    # 初期姿勢へ
    logger.info("[Step 5] 初期姿勢へ移動...")
    l_arm.set_start_state_to_current_state()
    l_arm.set_goal_state(configuration_name="l_arm_init_pose")
    plan_and_execute(robot, l_arm, logger, plan_params)

    r_arm.set_start_state_to_current_state()
    r_arm.set_goal_state(configuration_name="r_arm_init_pose")
    plan_and_execute(robot, r_arm, logger, plan_params)

    # グリッパを開く
    logger.info("[Step 5] グリッパ開放...")
    set_gripper(robot, l_gripper, robot_model, "l_gripper_group",
                GRIPPER_OPEN_L, logger, gripper_params)
    set_gripper(robot, r_gripper, robot_model, "r_gripper_group",
                GRIPPER_OPEN_R, logger, gripper_params)

    # プリグラスプ姿勢 (手首目標の上方にオフセット)
    Z_OFFSET = args.pre_grasp_z_offset
    lh_pre = lh_base.copy(); lh_pre[2] += Z_OFFSET
    rh_pre = rh_base.copy(); rh_pre[2] += Z_OFFSET

    logger.info(f"[Step 5] プリグラスプへ移動 (z+{Z_OFFSET:.2f}m)...")
    node.move_arm_to(robot, l_arm, logger, plan_params,
                     lh_pre, args.l_pose_link, q_down_l)
    node.move_arm_to(robot, r_arm, logger, plan_params,
                     rh_pre, args.r_pose_link, q_down_r)

    # グラスプ位置へ下降
    logger.info("[Step 5] グラスプ位置へ下降...")
    node.move_arm_to(robot, l_arm, logger, plan_params,
                     lh_base, args.l_pose_link, q_down_l)
    node.move_arm_to(robot, r_arm, logger, plan_params,
                     rh_base, args.r_pose_link, q_down_r)

    # グリッパを閉じる (把持)
    logger.info("[Step 5] グリッパ閉鎖 (把持)...")
    set_gripper(robot, l_gripper, robot_model, "l_gripper_group",
                GRIPPER_GRASP_L, logger, gripper_params)
    set_gripper(robot, r_gripper, robot_model, "r_gripper_group",
                GRIPPER_GRASP_R, logger, gripper_params)

    logger.info("[完了] 把持パイプライン終了")
    logger.info(f"  左手首 (base_link): {lh_base}")
    logger.info(f"  右手首 (base_link): {rh_base}")
    return lh_base, rh_base


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="sciurus17 両手把持パイプライン (ROS2)")

    # サーバ・メッシュ
    parser.add_argument("--server-url",   default="http://10.40.1.126:8080",
                        help="姿勢推定サーバ URL")
    parser.add_argument("--mesh-out",     default="meshes/object.ply",
                        help="メッシュ保存先 PLY ファイルパス")
    parser.add_argument("--mesh-method",  default="bpa",
                        choices=["bpa", "poisson", "knn"])

    # 物体選択
    parser.add_argument("--click-x", type=int, default=-1,
                        help="物体選択 x ピクセル (-1 でインタラクティブ選択)")
    parser.add_argument("--click-y", type=int, default=-1)

    # Shape2Gesture モデル
    parser.add_argument("--model-dir",   default="save_model",
                        help="Shape2Gesture モデルディレクトリ")
    parser.add_argument("--model-epoch", type=int, default=69)

    # カメラトピック
    parser.add_argument("--color-topic",
                        default="/sciurus17/camera/color/image_raw")
    parser.add_argument("--depth-topic",
                        default="/sciurus17/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic",
                        default="/sciurus17/camera/color/camera_info")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="深度値 → メートル変換係数 (uint16 → m)")

    # 座標フレーム
    parser.add_argument("--camera-frame", default="camera_color_optical_frame",
                        help="カメラ座標フレーム ID")
    parser.add_argument("--base-frame",   default="base_link",
                        help="ロボットベースフレーム ID")

    # MoveItPy
    parser.add_argument("--l-arm-group",  default="l_arm_group",
                        help="左腕 MoveIt 計画グループ名")
    parser.add_argument("--r-arm-group",  default="r_arm_group",
                        help="右腕 MoveIt 計画グループ名")
    parser.add_argument("--l-pose-link",  default="l_link7",
                        help="左腕エンドエフェクタリンク名")
    parser.add_argument("--r-pose-link",  default="r_link7",
                        help="右腕エンドエフェクタリンク名")
    parser.add_argument("--pre-grasp-z-offset", type=float, default=0.10,
                        help="プリグラスプ姿勢の上方オフセット [m]")

    args = parser.parse_args()

    rclpy.init()
    node = GraspPipelineNode(args)

    # カメラデータと TF を受信するためのスピンスレッド
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_pipeline(node, args)
    except TimeoutError as e:
        node.get_logger().error(f"タイムアウト: {e}")
        sys.exit(1)
    except Exception as e:
        node.get_logger().error(f"パイプラインエラー: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
