#!/bin/bash
# demo.launch.py を起動し、準備完了後に eef_position_tracker.py を自動起動する

source /ros2_ws/install/setup.bash

echo "[ビルド] sciurus17_moveit_config をビルド中..."
cd /ros2_ws && colcon build --packages-select sciurus17_moveit_config --symlink-install 2>/dev/null
source /ros2_ws/install/setup.bash

echo "[起動] demo.launch.py を開始..."
ros2 launch sciurus17_examples demo.launch.py \
    use_mock_components:=true \
    use_head_camera:=false \
    use_chest_camera:=false \
    rviz_config:=/sciurus17_m2/sciurus17/range_check.rviz &
LAUNCH_PID=$!

echo "[待機] /joint_states トピックが来るまで待ちます..."
until ros2 topic list 2>/dev/null | grep -q "/joint_states"; do
    sleep 1
done

echo "[待機] move_group が起動するまで待ちます..."
until ros2 node list 2>/dev/null | grep -q "move_group"; do
    sleep 1
done

# move_group の初期化完了まで少し余裕を持つ
sleep 5

echo "[起動] eef_position_tracker.py を開始..."
python3 -u /sciurus17_m2/ros/eef_position_tracker.py 2>&1

# eef_position_tracker が終了したら launch も止める
kill $LAUNCH_PID 2>/dev/null
wait $LAUNCH_PID 2>/dev/null
