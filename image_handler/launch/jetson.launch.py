# jetson.launch.py
import launch
from launch_ros.actions import Node

def generate_launch_description():
    camera_manager = Node(
        package='image_handler',
        executable='camera_manager',
        name='camera_manager',
        output='screen',
    )

    image_overlay = Node(
        package='image_handler',
        executable='image_overlay',
        name='image_overlay',
        output='screen',
    )

    return launch.LaunchDescription([
        camera_manager,
        image_overlay,
    ])