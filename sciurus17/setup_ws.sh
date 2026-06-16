#!/bin/bash
# コンテナ内で実行: ROS2 ワークスペースを初期セットアップする
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR=/ros2_ws

echo "=== ROS2 ワークスペースのセットアップを開始します ==="

# src/ が存在しない場合は作成
mkdir -p "${WS_DIR}/src"

# vcs import でパッケージをクローン
echo "--- パッケージをクローン中 ---"
cd "${WS_DIR}/src"
vcs import < "${SCRIPT_DIR}/sciurus17.repos"

# rosdep で依存関係をインストール
echo "--- 依存関係をインストール中 ---"
source /opt/ros/jazzy/setup.bash
cd "${WS_DIR}"
rosdep install --from-paths src --ignore-src -r -y

# ビルド
echo "--- colcon build 中 ---"
colcon build --symlink-install

echo "=== セットアップ完了 ==="
echo "次回から: source /ros2_ws/install/setup.bash"
