#!/bin/bash
# demo.launch.py + eef_position_tracker を一発起動するスクリプト
# 使い方:
#   docker compose run --rm sciurus17 bash /sciurus17_m2/ros/run_eef_tracker.sh

set -e

source /ros2_ws/install/setup.bash

# demo をバックグラウンドで起動
echo "=== demo.launch.py 起動中 ==="
ros2 launch sciurus17_examples demo.launch.py use_mock_components:=true use_head_camera:=false use_chest_camera:=false &
DEMO_PID=$!

# move_group ノードが現れるまで待機
echo "=== move_group の起動を待機中 ==="
until ros2 node list 2>/dev/null | grep -q "move_group"; do
    sleep 1
done
echo "=== move_group 検出 — さらに 8秒待機 (DDS安定化) ==="
sleep 8

# eef_position_tracker 起動
echo "=== eef_position_tracker 起動 ==="
python3 /sciurus17_m2/ros/eef_position_tracker.py

# tracker 終了後に demo も停止
kill $DEMO_PID 2>/dev/null
wait $DEMO_PID 2>/dev/null
echo "=== 終了 ==="
