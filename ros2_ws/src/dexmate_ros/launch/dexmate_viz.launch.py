"""
Launch dexmate services + RViz.

After launch, open another terminal and run:
    ros2 run dexmate_ros interactive_viz --ros-args \\
        -p rgb_path:=/path/to/rgb.jpg \\
        -p depth_path:=/path/to/depth.png \\
        -p api_key:=YOUR_KEY
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("dexmate_ros")
    rviz_config = os.path.join(pkg_share, "config", "dexmate.rviz")

    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("box_threshold", default_value="0.25"),
        DeclareLaunchArgument("model", default_value="gemini-2.5-flash"),
        DeclareLaunchArgument("default_api_key", default_value=""),

        LogInfo(msg="Starting dexmate services + RViz ..."),
        LogInfo(msg="Run interactive_viz in another terminal to start."),

        Node(
            package="dexmate_ros",
            executable="scene_node",
            name="scene_understanding_node",
            output="screen",
            parameters=[{
                "device": LaunchConfiguration("device"),
                "box_threshold": LaunchConfiguration("box_threshold"),
            }],
        ),
        Node(
            package="dexmate_ros",
            executable="planner_node",
            name="task_planner_node",
            output="screen",
            parameters=[{
                "model": LaunchConfiguration("model"),
                "default_api_key": LaunchConfiguration("default_api_key"),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="log",
        ),
    ])
