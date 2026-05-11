"""
Shape2Gesture 把持姿勢生成サーバ

GraspGenerator (Shape2Gesture) を HTTP サービスとして提供する。
SAM-6D サーバ (server.py) と独立して動作する。

起動方法:
    python server_grasp.py \
        --grasp-model-dir /path/to/save_model \
        --grasp-client-dir /path/to/client \
        --host 0.0.0.0 --port 8082

エンドポイント:
    GET  /health          — 死活確認
    POST /generate_grasp  — PLY メッシュから把持姿勢生成
"""

import argparse
import os
import sys
import threading

import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Shape2Gesture Grasp Generation Server")

_grasp_model_dir: str = ""
_grasp_client_dir: str = ""
_grasp_generator = None
_grasp_generator_lock = threading.Lock()

# /reconstruct_mesh が返す Docker パス → ホストパスのマッピング (server.py と共有 tmp)
_host_tmp: str   = os.path.join(_SERVER_DIR, "tmp")
_docker_tmp: str = "/workspace/tmp"


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, _SERVER_DIR)
    except ValueError:
        return path


def _align_from_gravity(pts: np.ndarray, gravity_cam=None):
    """点群を重力方向に揃える (Y-down)。整列済み点群と回転行列 R_corr を返す。"""
    if gravity_cam is None or np.linalg.norm(gravity_cam) < 1e-6:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        g = gravity_cam / np.linalg.norm(gravity_cam)
        target = np.array([0.0, -1.0, 0.0])
        v = np.cross(g, target)
        s = float(np.linalg.norm(v))
        c = float(np.dot(g, target))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]], dtype=np.float64)
            R = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s ** 2)
    R = np.asarray(R, dtype=np.float64)
    return (R @ pts.T).T.astype(pts.dtype), R


def _get_grasp_generator():
    """GraspGenerator を遅延ロードして返す。model_dir 未指定なら None。"""
    global _grasp_generator
    if _grasp_generator is not None:
        return _grasp_generator
    if not _grasp_model_dir:
        return None
    with _grasp_generator_lock:
        if _grasp_generator is None:
            if _grasp_client_dir and _grasp_client_dir not in sys.path:
                sys.path.insert(0, _grasp_client_dir)
            from pipeline.grasp_generator import GraspGenerator
            gen = GraspGenerator(model_dir=_grasp_model_dir)
            gen.load_models()
            _grasp_generator = gen
            print(f"[GraspServer] GraspGenerator ロード完了 (model_dir={_grasp_model_dir})")
    return _grasp_generator


@app.get("/health")
def health():
    gen = _grasp_generator  # ロード済みかどうかのみ確認
    return {"status": "ok", "model_loaded": gen is not None,
            "model_dir": _grasp_model_dir}


@app.post("/generate_grasp")
async def generate_grasp(
    mesh_path: str = Form(...),
    gravity_x: float = Form(0.0),
    gravity_y: float = Form(0.0),
    gravity_z: float = Form(0.0),
    num_samples: int = Form(1),
):
    """
    Shape2Gesture で把持姿勢を生成する

    Args:
        mesh_path:       サーバ側の PLY ファイルパス
                         (/reconstruct_mesh が返す mesh_path をそのまま渡す)
        gravity_x/y/z:   カメラ座標系の重力方向ベクトル (0,0,0=未指定→固定補正)
        num_samples:     生成する把持候補数

    Returns:
        {
          "grasps": [
            {"left_hand": [[x,y,z],...23関節],
             "right_hand": [[x,y,z],...23関節]}
          ],
          "mesh_scale_m": float,   # 正規化座標→メートルのスケール係数
          "R_corr": [[...]]        # 重力アライメント回転行列 (3x3)
        }

    クライアント側での座標変換:
        R = (R_from_sam6d @ R_corr.T)
        pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)
        wrist_cam = normalized_to_camera(left_hand[0], pose)
    """
    gen = _get_grasp_generator()
    if gen is None:
        raise HTTPException(503, "GraspGenerator が未ロードです。--grasp-model-dir を指定して起動してください。")

    # Docker パス → ホストパスに変換 (server.py の tmp を共有)
    mesh_host = mesh_path.replace(_docker_tmp, _host_tmp)
    if not os.path.exists(mesh_host):
        raise HTTPException(404, f"メッシュが見つかりません: {mesh_host}")

    # PLY 点群読み込み (open3d があれば使用、なければ plyfile)
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(mesh_host)
        mesh_pts = np.asarray(pcd.points, dtype=np.float32)
        if len(mesh_pts) == 0:
            raise ValueError("open3d で点群が空")
    except Exception:
        from plyfile import PlyData
        ply_data = PlyData.read(mesh_host)
        v = ply_data["vertex"]
        mesh_pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if len(mesh_pts) == 0:
        raise HTTPException(500, "点群が空です")

    print(f"[GraspServer] PLY 読み込み: {len(mesh_pts)} pts ({_rel(mesh_host)})")

    # 重力アライメント
    gvec = np.array([gravity_x, gravity_y, gravity_z], dtype=np.float32)
    gravity_cam = gvec if float(np.linalg.norm(gvec)) > 1e-6 else None
    mesh_pts_aligned, R_corr = _align_from_gravity(mesh_pts, gravity_cam)

    # メッシュスケール: mm 単位の点群 → 最大半径 [m]
    centered = mesh_pts_aligned - mesh_pts_aligned.mean(axis=0)
    mesh_scale_m = float(np.max(np.linalg.norm(centered, axis=1))) / 1000.0

    # 把持姿勢生成
    print(f"[GraspServer] 生成中 (num_samples={num_samples}, scale={mesh_scale_m:.4f} m)...")
    results = gen.generate(mesh_pts_aligned, num_samples=num_samples)

    grasps = [
        {"left_hand": lh.tolist(), "right_hand": rh.tolist()}
        for lh, rh in results
    ]
    print(f"[GraspServer] 完了: {len(grasps)} grasps")

    return JSONResponse({
        "grasps":       grasps,
        "mesh_scale_m": mesh_scale_m,
        "R_corr":       R_corr.tolist(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-model-dir",
                        default=os.path.join(os.path.dirname(_SERVER_DIR), "client", "save_model"),
                        help="Shape2Gesture の save_model ディレクトリ")
    parser.add_argument("--grasp-client-dir",
                        default=os.path.join(os.path.dirname(_SERVER_DIR), "client"),
                        help="client/ ディレクトリ (pipeline/models のインポート元)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host-tmp", default=os.path.join(_SERVER_DIR, "tmp"))
    parser.add_argument("--docker-tmp", default="/workspace/tmp")
    args = parser.parse_args()

    _grasp_model_dir = args.grasp_model_dir
    _grasp_client_dir = args.grasp_client_dir
    _host_tmp   = args.host_tmp
    _docker_tmp = args.docker_tmp

    print("=" * 50)
    print("  Shape2Gesture Grasp Generation Server")
    print(f"  host:port:       {args.host}:{args.port}")
    print(f"  grasp_model_dir: {_grasp_model_dir}")
    print(f"  grasp_client_dir:{_grasp_client_dir}")
    print(f"  host_tmp:        {_host_tmp}")
    print("=" * 50)

    # 起動時にモデルをロード
    _get_grasp_generator()

    uvicorn.run(app, host=args.host, port=args.port)
