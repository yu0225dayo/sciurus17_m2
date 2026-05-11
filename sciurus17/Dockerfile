FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libpcl-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    udev \
    python3-tk \
    fonts-noto-cjk \
    ros-jazzy-hardware-interface \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-gripper-controllers \
    ros-jazzy-rt-manipulators-cpp \
    ros-jazzy-moveit \
    ros-jazzy-moveit-py \
    ros-jazzy-moveit-ros-planning-interface \
    ros-jazzy-moveit-ros-move-group \
    ros-jazzy-moveit-kinematics \
    ros-jazzy-moveit-planners \
    ros-jazzy-moveit-simple-controller-manager \
    ros-jazzy-moveit-configs-utils \
    ros-jazzy-moveit-ros-visualization \
    ros-jazzy-moveit-setup-assistant \
    ros-jazzy-pcl-ros \
    ros-jazzy-usb-cam \
    ros-jazzy-realsense2-camera \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros-gz \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir Pillow "opencv-python-headless>=4.8" numpy requests plyfile

WORKDIR /ros2_ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
