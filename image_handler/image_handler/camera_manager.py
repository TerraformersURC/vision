# camera_manager.py
import subprocess
import signal
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String


ZED_CONFIGS = {
    0: {
        'camera_name': 'zed0',
        'camera_model': 'zed2i',
        'serial_number': 35849989,
        'node_namespace': '/zed0',
    },
    1: {
        'camera_name': 'zed1',
        'camera_model': 'zed2i',
        'serial_number': 32050746,
        'node_namespace': '/zed1',
    },
}


class CameraManager(Node):
    def __init__(self):
        super().__init__('camera_manager')

        self.active_process = None
        self.active_camera = None
        self.switching = False

        self.active_cam_pub = self.create_publisher(String, '/active_camera_ns', 10)
        self.create_subscription(Int32, '/camera_select', self.on_camera_select, 10)

        self._launch_camera(0)

    def on_camera_select(self, msg: Int32):
        target = msg.data
        if target == self.active_camera:
            return
        if target not in ZED_CONFIGS:
            self.get_logger().warn(f'Invalid camera index: {target}')
            return
        if self.switching:
            self.get_logger().warn('Already switching, ignoring request.')
            return

        self.switching = True
        self.get_logger().info(f'Switching to camera {target}')
        self._kill_current()
        self._launch_camera(target)

    def _launch_camera(self, camera_id: int):
        cfg = ZED_CONFIGS[camera_id]

        cmd = [
            'ros2', 'launch', 'zed_wrapper', 'zed_camera.launch.py',
            f'camera_model:={cfg["camera_model"]}',
            f'camera_name:={cfg["camera_name"]}',
            f'serial_number:={cfg["serial_number"]}',
        ]

        self.get_logger().info(f'Launching camera {camera_id}: {" ".join(cmd)}')
        self.active_process = subprocess.Popen(cmd)
        self.active_camera = camera_id
        self.switching = False

        # Delay publishing until camera is initialized
        topic = f'{cfg["node_namespace"]}/zed_node/rgb/color/rect/image'
        self.create_timer(4.0, lambda: self._publish_active(topic))

    def _publish_active(self, topic: str):
        msg = String()
        msg.data = topic
        self.active_cam_pub.publish(msg)
        self.get_logger().info(f'Published active camera topic: {topic}')

    def _kill_current(self):
        if self.active_process:
            self.get_logger().info('Killing current camera process...')
            self.active_process.send_signal(signal.SIGINT)
            try:
                self.active_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.active_process.kill()
            self.active_process = None
            self.active_camera = None


def main(args=None):
    rclpy.init(args=args)
    node = CameraManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()