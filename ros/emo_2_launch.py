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
            executable='emo_node_2',
            name='emotion_recognition_node_2',
            parameters=[{
                'video_topic': '/camera/color/image_raw',
                'audio_topic': '/microphone/audio_data',
                'emotion_topic': '/robot/detected_emotion',
                'model_size': 'small',
                'weights_path': '/home/franc/ros2-ws/src/emotion_pkg/emotion_pkg/final_model_1s.pt',
                'record': False,
                'output_dir': '/tmp/robot_recordings',
                'enable_audio': True,
                'enable_video': True,
                'audio_energy_threshold': 0.01,
                'face_visibility_threshold': 0.5,
                'confidence_threshold': 0.5,
            }],
            output='screen'
        )
    ])
