# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**sciurus17**（双腕人型ロボット）を動かすプロジェクト。

sciurus17 の主な機能:
- **カメラキャプチャ**: 胸部 RealSense カメラで RGBD 撮影
- **計算機への送信**: GPU サーバへ RGBD を HTTP 送信し物体姿勢推定
- **把持姿勢生成**: Shape2Gesture で両手把持姿勢を計算
- **アームの移動実行**: MoveItPy で左右アームを目標手首位置へ移動

```
sciurus17 (ROS2 PC)                  GPU 計算機 (10.40.1.126)
  RealSense RGBD カメラ
    └─ sciurus17_grasp_pipeline.py
         │  RGBD ──HTTP:8080──────→  server/server.py  (SAM2 + SAM-3D)
         │                                ↓ HTTP:8081
         │                           SAM-6D Docker
         │  ←── R, t (6DoF pose) ──────────────────────
         │
         ├─ GraspGenerator (Shape2Gesture) → hand[0] (手首座標・物体系)
         ├─ TF2 変換 (camera_frame → base_link)
         └─ MoveItPy → 左右アーム移動
```

## 起動手順

### GPU 計算機側

```bash
# SAM-6D Docker を起動
cd ~/ws/project/server
docker compose up -d sam6d
curl http://localhost:8081/health

# server.py を起動
python server.py \
    --sam-checkpoint /ws/okada/project/sam_vit_h_4b8939.pth \
    --sam3d-config   ~/ws/sam-3d-objects/checkpoints/hf/pipeline.yaml \
    --sam3d-repo     ~/ws/sam-3d-objects \
    --sam6d-service  http://localhost:8081 \
    --host 0.0.0.0 --port 8080
```

### sciurus17 PC 側（ROS2 環境）

```bash
# フルパイプライン: カメラ → 計算機送信 → 両手把持 → アーム移動
python3 ros/sciurus17_grasp_pipeline.py \
    --server-url http://10.40.1.126:8080 \
    --mesh-out meshes/object.ply \
    --click-x 320 --click-y 240

# インタラクティブ選択（ウィンドウでクリック）
python3 ros/sciurus17_grasp_pipeline.py \
    --server-url http://10.40.1.126:8080

# PC クライアントからテスト（RealSense 直接接続）
cd client
python test_demo.py --data_path saved_data/test_YYYYMMDD_HHMMSS
python main.py
```

## ディレクトリ構成

| ディレクトリ | 内容 |
|---|---|
| `ros/` | sciurus17 上で動く ROS ノード群 |
| `client/` | PC からのテスト・デバッグ用クライアント |
| `server/` | GPU 計算機上の姿勢推定サーバ |
| `sciurus17/` | sciurus17_ros (ROS2 パッケージ, submodule) |

## アーキテクチャ詳細

### sciurus17 把持パイプライン (`ros/sciurus17_grasp_pipeline.py`)

ROS2 ノード。sciurus17 PC 上で実行する統合パイプライン。

| ステップ | 処理 |
|---|---|
| Step 0 | ROS トピックから RGBD 取得 + カメラパラメータ受信 |
| Step 1 | HTTP: SAM-3D でメッシュ生成 (`/reconstruct_mesh`) |
| Step 2 | HTTP: SAM-6D で 6DoF pose 推定 (`/pose_estimate`) → R, t |
| Step 3 | Shape2Gesture で両手把持姿勢生成 → `hand[0]` (手首、物体正規化座標) |
| Step 4 | TF2 変換: `camera_color_optical_frame` → `base_link` |
| Step 5 | MoveItPy: 左右アームをプリグラスプ → グラスプ位置へ移動、グリッパ開閉 |

### ROS トピック

| トピック | 型 | 用途 |
|---|---|---|
| `/sciurus17/camera/color/image_raw` | `sensor_msgs/Image` | カラー画像入力 |
| `/sciurus17/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 深度画像入力 |
| `/sciurus17/camera/color/camera_info` | `sensor_msgs/CameraInfo` | カメラ内部パラメータ |

### 座標変換フロー

```
Shape2Gesture 出力: hand[0] (23関節のindex 0 = 手首) — 物体正規化座標系
    ↓ normalized_to_camera(pose)  [client/utils/coord_transform.py]
カメラ座標系 [m]  (frame: camera_color_optical_frame)
    ↓ TF2 tf_buffer.transform()
ロボット座標系 [m]  (frame: base_link)
    ↓ MoveItPy set_goal_state → execute
アーム目標位置
```

### サーバ (`server/server.py`)

- `POST /reconstruct_mesh` — RGB → SAM2 マスク → SAM-3D → PLY 返却
- `POST /pose_estimate` — RGB + depth → SAM2 マスク → SAM-6D (Docker) → R, t 返却
- SAM-3D はオンデマンドロード（推論後に即削除して GPU 解放）
- SAM-6D は Docker コンテナ (sam6d_service) へ HTTP プロキシ

### MoveItPy 計画グループ

| グループ | リンク先 | 用途 |
|---|---|---|
| `l_arm_group` | `l_link7` | 左腕制御 |
| `r_arm_group` | `r_link7` | 右腕制御 |
| `l_gripper_group` | — | 左グリッパ |
| `r_gripper_group` | — | 右グリッパ |

### クライアント (`client/`) — デバッグ・テスト用

| モジュール | 役割 |
|---|---|
| `main.py` | RealSense 直接接続 + キーボード操作 UI |
| `test_demo.py` | 保存済み RGBD でテスト |
| `pipeline/sam6d_detector.py` | サーバ HTTP クライアント (`SAM6DClient`) |
| `pipeline/grasp_generator.py` | Shape2Gesture 把持姿勢生成 |
| `utils/coord_transform.py` | 座標変換・`CameraIntrinsics` / `ObjectPose` |

### ROS スクリプト (`ros/`)

| スクリプト | 役割 |
|---|---|
| `sciurus17_gui.py` | **メイン GUI**: 全工程をボタンで操作 (ROS2 + tkinter) |
| `sciurus17_grasp_pipeline.py` | CUI 版パイプライン: コマンドライン引数で全工程 (ROS2) |
| `rgbd_server.py` | ROS1: RGBD HTTP 提供 + 手首座標受信パブリッシュ |
| `okada_arm.py` | ROS1: 単腕把持テスト |
| `cai_getIMAGEandPUBLISH2.py` | ROS1: 動画録画 + 首制御 |

## 既知の問題

- **スケール不一致**: SAM-3D が生成する PLY のスケールが実寸と合わず SAM-6D の姿勢推定精度が低下する。`server.py` 内に深度から推定サイズでスケール補正するコードがあるが根本解決には至っていない。
- **把持座標の精度**: Shape2Gesture の出力 hand[0] は物体正規化座標系なので、スケールと TF 変換の精度が把持成功率に直結する。
- 代替パイプライン (`~/ws/project-fp/`) で FoundationPose + 深度直接 BPA メッシュを検証中。

## 共有ディレクトリ

- `tmp/` — サーバ・Docker 間の中間ファイル共有（Docker 内では `/workspace/tmp/`）
- `server/SAM-6D/` — SAM-6D リポジトリ（submodule）
- `server/sam-3d-objects/` — SAM-3D リポジトリ（submodule）
- `sciurus17/ws/src/sciurus17_ros/` — sciurus17 ROS2 パッケージ（submodule, jazzy ブランチ）
