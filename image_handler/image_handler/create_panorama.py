import time
import cv2
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PanoramaPublisher(Node):
    def __init__(self):
        super().__init__('panorama_publisher')
        self.bridge = CvBridge()

        qos_pub = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher_ = self.create_publisher(Image, 'panorama', qos_pub)

        # Match the ZED wrapper's QoS
        qos_sub = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.create_subscription(
            Image,
            '/zed0/zed_node/left/image_rect_color', # Change "zed0" if using a different camera / camera name
            self.image_callback,
            qos_sub
        )
        self.latest_frame = None

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def capture_frames(self, num_images=3, delay_sec=1.5):
        captured = []
        i = 0
        while i < num_images:
            rclpy.spin_once(self, timeout_sec=1.0)
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                filename = f"capture_{i}.png"
                cv2.imwrite(filename, frame)
                captured.append(frame)
                self.get_logger().info(f"Captured {filename}")
                i += 1
                if i < num_images:
                    time.sleep(delay_sec)
            else:
                self.get_logger().warn("Waiting for image from ZED wrapper...")
                time.sleep(0.1)
        return captured

    def publish_panorama(self, pano_bgr):
        msg = self.bridge.cv2_to_imgmsg(pano_bgr, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'zed2i_left_camera_optical_frame'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published panorama to /panorama')
    
def stitch_panorama(images, output_path="panorama.png"):
    if len(images) < 2:
        print("Need at least 2 images to stitch a panorama.")
        return None

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, pano)
        print(f"Panorama saved to {output_path}")
        return pano
    else:
        print(f"Stitching failed with status code: {status}")
        return None

def main():
    rclpy.init()
    node = PanoramaPublisher()

    images = node.capture_frames()
    pano = stitch_panorama(images)

    if pano is not None:
        node.publish_panorama(pano)
        rclpy.spin_once(node, timeout_sec=1.0)
    else:
        node.get_logger().error("Stitching failed, nothing published.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()