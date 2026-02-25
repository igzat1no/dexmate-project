"""Launch both dexmate service nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("box_threshold", default_value="0.25"),
        DeclareLaunchArgument("model", default_value="gemini-2.5-flash"),
        DeclareLaunchArgument("default_api_key", default_value=""),

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
    ])
