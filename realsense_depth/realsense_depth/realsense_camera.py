import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import rclpy
 
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

FPS = 30

class RealsenseCam():
    def __init__(self) -> None:
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, *(640, 480), rs.format.z16, FPS)
        self.config.enable_stream(rs.stream.color, *(1920, 1080), rs.format.bgr8, FPS)

        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

    def get_depth_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        depth_frame = np.asarray(frames.get_depth_frame().get_data())
        return depth_frame

    def get_depth_colormap(self) -> np.ndarray:
        depth_colormap = cv.applyColorMap(
            cv.convertScaleAbs(
                self.get_depth_frame(),
                alpha=0.03
            ),
            cv.COLORMAP_JET
        )
        return depth_colormap

    def get_color_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = np.asarray(frames.get_color_frame().get_data())
        return color_frame

class RealsensePublisher(Node):
    def __init__(self):
        super().__init__("realsense_camera_publisher")
        
        self.depth_pub = self.create_publisher(Image, "/depth_image", 10)
        self.color_pub = self.create_publisher(Image, "/color_image", 10)        
        
        self.create_timer(1/FPS, self.send_frames)
        self.bridge = CvBridge()

        self.cam = RealsenseCam()

    def send_frames(self):
        depth_frame = self.cam.get_depth_frame()
        color_frame = self.cam.get_color_frame()

        depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, "mono16")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

        color_msg = self.bridge.cv2_to_imgmsg(color_frame, "bgr8")
        color_msg.header.stamp = self.get_clock().now().to_msg()
        self.color_pub.publish(color_msg)
    
    def destroy_node(self):
        self.cam.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealsensePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    