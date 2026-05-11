"""
SAM-6D クライアントモジュール (ローカルPC側)

サーバとの通信:
    オフライン (物体ごとに1回):
        RGB → /reconstruct_mesh → reference mesh (.ply) 保存
        サーバが SAM-3D でメッシュ生成 + SAM-6D でテンプレートレンダリング

    オンライン (毎フレーム):
        RGB + depth → /pose_estimate → 6DoF pose (R, t) 取得
        サーバが SAM-6D でポーズ推定
"""

import os
import numpy as np
import cv2
import requests
from typing import Tuple, Optional

from utils.coord_transform import CameraIntrinsics


class SAM6DClient:
    """
    SAM-6D サーバへの HTTP クライアント

    使用例 (オフライン):
        client = SAM6DClient(server_url="http://10.40.1.126:8080")
        client.save_reference_mesh(rgb, "meshes/cup.ply")

    使用例 (オンライン):
        client.load_reference_mesh("meshes/cup.ply")
        R, t = client.estimate_pose(rgb, depth, intrinsics)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout_mesh: float = 300.0,
        timeout_pose: float = 30.0,
        grasp_server_url: Optional[str] = None,
        timeout_grasp: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_mesh = timeout_mesh
        self.timeout_pose = timeout_pose
        self.grasp_server_url = grasp_server_url.rstrip("/") if grasp_server_url else None
        self.timeout_grasp = timeout_grasp
        self._mesh_path: Optional[str] = None      # ローカルの .ply パス
        self._server_mesh_path: str = ""            # サーバ側の .ply パス
        self._template_dir: str = ""               # サーバ側テンプレートディレクトリ
        self._object_size_mm: float = 0.0          # ユーザ指定サイズ (0=自動)

    # ------------------------------------------------------------------
    # オフライン: reference mesh の生成・保存
    # ------------------------------------------------------------------

    def save_reference_mesh(
        self,
        rgb: np.ndarray,
        mesh_save_path: str,
        click_x: int = -1,
        click_y: int = -1,
        seed: int = 42,
        mesh_method: str = "bpa",
        object_size_mm: float = 0.0,
    ) -> Tuple[str, list, list]:
        """
        RGB画像をサーバに送り、SAM-3D で生成した reference mesh を保存する

        サーバは SAM-3D でメッシュ生成後、SAM-6D でテンプレートをレンダリングし
        テンプレートディレクトリのパスをヘッダで返す。

        Args:
            rgb:             (H, W, 3) BGR 画像
            mesh_save_path:  ローカルの保存先 .ply パス
            click_x, click_y: SAM プロンプト座標 (-1,-1 で画像中央)
            seed:            SAM-3D シード値

        Returns:
            保存した .ply のパス
        """
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = buf.tobytes()

        print(f"[SAM6D] reference mesh 生成中 (プロンプト: ({click_x},{click_y}))...")
        resp = requests.post(
            f"{self.server_url}/reconstruct_mesh",
            files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            data={"click_x": click_x, "click_y": click_y, "seed": seed, "mesh_method": mesh_method,
                  "object_size_mm": object_size_mm},
            timeout=self.timeout_mesh,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"サーバエラー ({resp.status_code}): {resp.text}")

        import base64
        data = resp.json()

        # PLY を保存
        os.makedirs(os.path.dirname(os.path.abspath(mesh_save_path)), exist_ok=True)
        with open(mesh_save_path, "wb") as f:
            f.write(base64.b64decode(data["ply_b64"]))

        # 3マスクをデコード
        masks = []
        for b64 in data.get("masks_b64", []):
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            masks.append(cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE))
        scores   = data.get("scores", [])
        best_idx = data.get("best_idx", 0)

        self._server_mesh_path = data.get("mesh_path", "")
        self._template_dir     = data.get("template_dir", "")
        mask_u = data.get("mask_center_u", "?")
        mask_v = data.get("mask_center_v", "?")

        print(f"[SAM6D] mesh 保存完了: {mesh_save_path}  mask_center=({mask_u},{mask_v})")
        print(f"[SAM6D] SAM scores: {[f'{s:.3f}' for s in scores]}  best_idx={best_idx}")
        if self._server_mesh_path:
            print(f"[SAM6D] サーバ側 mesh: {self._server_mesh_path}")
        if self._template_dir:
            print(f"[SAM6D] テンプレート: {self._template_dir}")

        self._mesh_path = mesh_save_path
        self._object_size_mm = object_size_mm
        return mesh_save_path, masks, scores

    def save_reference_mesh_interactive(
        self,
        rgb: np.ndarray,
        mesh_save_path: str,
        seed: int = 42,
        mesh_method: str = "bpa",
        min_mask_ratio: float = 0.002,
    ) -> Tuple[str, int, int, list, list]:
        """インタラクティブモード: クリックして物体を指定する

        Args:
            min_mask_ratio: ベストマスクの面積がこの割合未満なら再選択を促す (デフォルト 0.2%)
        """
        img_area = rgb.shape[0] * rgb.shape[1]
        win_name = "Select Object (click + Enter)"

        while True:
            clicked = []

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked.clear()
                    clicked.append((x, y))

            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, mouse_callback)
            print("[SAM6D] 物体をクリックして選択し、Enter で確定してください。")

            while True:
                # ウィンドウが閉じられたら終了
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("ウィンドウが閉じられました。")
                display = rgb.copy()
                if clicked:
                    cv2.circle(display, clicked[0], 8, (0, 255, 0), -1)
                    cv2.putText(display, "Press Enter to confirm", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(win_name, display)
                key = cv2.waitKey(1)
                if key == 13 and clicked:
                    break
                elif key == 27:
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("キャンセルされました。")

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cx, cy = clicked[0]

            # cv2 の Enter キーが stdin に残るのでフラッシュ
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except ImportError:
                import termios, sys
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

            mesh_path, masks, scores = self.save_reference_mesh(
                rgb, mesh_save_path,
                click_x=cx, click_y=cy, seed=seed,
                mesh_method=mesh_method,
                object_size_mm=0.0,
            )

            # マスクサイズチェック — 小さすぎる場合は再選択
            if masks and scores:
                best_idx = int(np.argmax(scores))
                best_mask = masks[best_idx]
                mask_area = int(np.count_nonzero(best_mask))
                ratio = mask_area / img_area
                print(f"[SAM6D] ベストマスク面積: {mask_area} px ({ratio*100:.2f}% of image)")
                if ratio < min_mask_ratio:
                    print(f"[SAM6D] マスクが小さすぎます ({ratio*100:.2f}% < {min_mask_ratio*100:.2f}%)。"
                          "物体をもう一度クリックしてください。")
                    # マスクを重ねて警告表示
                    warn_disp = rgb.copy()
                    cv2.putText(warn_disp,
                                f"Mask too small ({ratio*100:.1f}%). Click again!",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(win_name, warn_disp)
                    cv2.waitKey(1500)
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    continue  # 再試行

            break  # チェック通過

        return mesh_path, cx, cy, masks, scores

    def load_reference_mesh(
        self,
        mesh_path: str,
        server_mesh_path: str = "",
        template_dir: str = "",
    ):
        """
        保存済みの reference mesh をセットする (セッション再開時)

        Args:
            mesh_path:        ローカルの .ply パス
            server_mesh_path: サーバ側の .ply パス
            template_dir:     サーバ側テンプレートディレクトリ
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"reference mesh が見つかりません: {mesh_path}")
        self._mesh_path        = mesh_path
        self._server_mesh_path = server_mesh_path
        self._template_dir     = template_dir
        print(f"[SAM6D] reference mesh ロード: {mesh_path}")

    # ------------------------------------------------------------------
    # オンライン: 6DoF pose 推定
    # ------------------------------------------------------------------

    def estimate_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        click_x: int = -1,
        click_y: int = -1,
        gravity: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RGBD + reference mesh からカメラ座標系での物体 6DoF pose を推定する

        サーバは SAM-6D (Docker) で推定し R, t を返す。
        メッシュファイルはサーバ側に保存済みのパスを指定するため再送不要。

        Args:
            rgb:        (H, W, 3) BGR 画像
            depth:      (H, W) float32 深度画像 [m]
            intrinsics: カメラ内部パラメータ

        Returns:
            R: (3, 3) float32 回転行列 (物体座標系 → カメラ座標系)
            t: (3,)   float32 平行移動ベクトル [m]
        """
        if not self._server_mesh_path:
            raise RuntimeError(
                "サーバ側 mesh パスが未設定です。"
                "save_reference_mesh() を実行するか、"
                "load_reference_mesh(server_mesh_path=...) で指定してください。"
            )

        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rgb_bytes   = buf.tobytes()
        depth_bytes = depth.astype(np.float32).tobytes()

        print("[SAM6D] 6DoF pose 推定中...")
        resp = requests.post(
            f"{self.server_url}/pose_estimate",
            files={
                "rgb_image":   ("frame.jpg", rgb_bytes,   "image/jpeg"),
                "depth_image": ("depth.bin", depth_bytes, "application/octet-stream"),
            },
            data={
                "fx":           intrinsics.fx,
                "fy":           intrinsics.fy,
                "cx":           intrinsics.cx,
                "cy":           intrinsics.cy,
                "mesh_path":       self._server_mesh_path,
                "template_dir":    self._template_dir,
                "click_x":         click_x,
                "click_y":         click_y,
                "object_size_mm":  self._object_size_mm,
                **( {"gravity_x": float(gravity[0]),
                     "gravity_y": float(gravity[1]),
                     "gravity_z": float(gravity[2])}
                    if gravity is not None else {} ),
            },
            timeout=self.timeout_pose,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"サーバエラー ({resp.status_code}): {resp.text}")

        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"pose 推定失敗: {data.get('error')}")

        R = np.array(data["R"], dtype=np.float32)  # (3, 3)
        t = np.array(data["t"], dtype=np.float32)  # (3,)
        print(f"[SAM6D] pose 推定完了: t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")

        # サーバ生成画像をデコードして返す
        import base64, cv2 as _cv2
        img_pose_bgr: Optional[np.ndarray] = None
        img_mesh_bgr: Optional[np.ndarray] = None
        for key, attr in [("img_pose", "pose"), ("img_mesh", "mesh")]:
            b64 = data.get(key, "")
            if b64:
                img = _cv2.imdecode(
                    np.frombuffer(base64.b64decode(b64), np.uint8),
                    _cv2.IMREAD_COLOR,
                )
                if key == "img_pose":
                    img_pose_bgr = img
                else:
                    img_mesh_bgr = img

        return R, t, img_pose_bgr, img_mesh_bgr

    # ------------------------------------------------------------------
    # 把持姿勢生成 (server_grasp.py へのリクエスト)
    # ------------------------------------------------------------------

    def generate_grasp(
        self,
        gravity: Optional[np.ndarray] = None,
        num_samples: int = 1,
    ) -> dict:
        """
        server_grasp.py に把持姿勢生成を依頼する

        Args:
            gravity:     カメラ座標系の重力方向ベクトル (None=固定補正)
            num_samples: 生成する把持候補数

        Returns:
            {
              "grasps": [{"left_hand": (23,3), "right_hand": (23,3)}, ...],
              "mesh_scale_m": float,
              "R_corr": (3,3) ndarray
            }

        使用例 (クライアント側での座標変換):
            result = client.generate_grasp(gravity=gvec, num_samples=1)
            lh_norm = result["grasps"][0]["left_hand"]   # (23,3)
            R = R_from_sam6d @ result["R_corr"].T
            pose = ObjectPose(center_3d=t, scale=result["mesh_scale_m"], R=R)
            wrist_cam = normalized_to_camera(lh_norm, pose)[0]
        """
        if not self._server_mesh_path:
            raise RuntimeError(
                "サーバ側 mesh パスが未設定です。"
                "save_reference_mesh() を先に呼んでください。"
            )

        url = (self.grasp_server_url or self.server_url) + "/generate_grasp"
        data = {
            "mesh_path":   self._server_mesh_path,
            "num_samples": num_samples,
            **( {"gravity_x": float(gravity[0]),
                 "gravity_y": float(gravity[1]),
                 "gravity_z": float(gravity[2])}
                if gravity is not None else {} ),
        }

        print(f"[SAM6D] 把持姿勢生成中 (grasp_server={url.split('/generate')[0]})...")
        resp = requests.post(url, data=data, timeout=self.timeout_grasp)

        if resp.status_code != 200:
            raise RuntimeError(f"GraspServer エラー ({resp.status_code}): {resp.text}")

        raw = resp.json()
        grasps = [
            {
                "left_hand":  np.array(g["left_hand"],  dtype=np.float32),
                "right_hand": np.array(g["right_hand"], dtype=np.float32),
            }
            for g in raw["grasps"]
        ]
        return {
            "grasps":       grasps,
            "mesh_scale_m": float(raw["mesh_scale_m"]),
            "R_corr":       np.array(raw["R_corr"], dtype=np.float32),
        }
