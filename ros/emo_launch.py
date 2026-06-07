from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='emotion_pkg',
            executable='camera_node',
            name='pc_camera_node'
        ),
        Node(
            package='emotion_pkg',
            executable='mic_node',
            name='pc_microphone_node'
        ),
        Node(
            package='emotion_pkg',
            executable='emotion_node',
            name='emotion_recognition_node',
            output='screen'
        )
    ])
