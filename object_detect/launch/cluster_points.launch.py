from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detect',
            executable='exclusion_zones',
            name='exclusion_zones',
            output='screen',
            parameters=[{'cluster_tolerance': 0.3}]
        )
    ])
