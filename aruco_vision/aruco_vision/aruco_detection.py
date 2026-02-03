import argparse
import time
import cv2
import sys
import math
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from aruco_msgs.msg import Aruco



ARUCO_TOPIC = "/aruco_topic"
VIDEO_TOPIC = "/robot/depth_camera/image_raw"


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        self.ros_cv_bridge = CvBridge()
        self.video_feed = None

        self.camera_matrix = None
        self.dist_coeffs = None

        #STORE ARUCO INFORMATION
        package_path = get_package_share_directory('aruco_vision')
        npz_path = os.path.join(package_path, 'config', 'camera_calibration_parameters.npz')


        with np.load(npz_path) as data:
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']

        self.aruco_publisher = self.create_publisher(Aruco, ARUCO_TOPIC, 10)
        self.camera_subscription = self.create_subscription(Image,VIDEO_TOPIC, self.set_videofeed_callback,10)




        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
        args = vars(ap.parse_args())

        

        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }

        if ARUCO_DICT.get(args["type"], None) is None:
            print(f"[INFO] ArUCo tag of '{args['type']}' is not supported")
            sys.exit(0)

        print(f"[INFO] detecting '{args['type']}' tags...")
        self.arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
        self.arucoParams = cv2.aruco.DetectorParameters_create()

        print("[INFO] starting video stream...")

        self.timer = self.create_timer(0.03,self.detect_aruco)



    def set_videofeed_callback(self,msg):
        try:
            self.video_feed = self.ros_cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            print("error converting camera feed to cv2 feed")
            return
        
        

    def detect_aruco(self):
        tag_size = 0.1  # meters
        focal_length = self.camera_matrix[0, 0]


        if(self.video_feed is None):
            print("No camera feed found!")
            return
        
            
        (corners, ids, rejected) = cv2.aruco.detectMarkers(self.video_feed, self.arucoDict, parameters=self.arucoParams)
        
        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))

                tag_pixel_size = sum(np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)) / 4
                distance_pixel_method = (tag_size * focal_length) / tag_pixel_size * 1000

                obj_points = np.array([
                    [-tag_size / 2, tag_size / 2, 0],
                    [ tag_size / 2, tag_size / 2, 0],
                    [ tag_size / 2, -tag_size / 2, 0],
                    [-tag_size / 2, -tag_size / 2, 0]
                ], dtype=np.float32)

                retval, rvec, tvec = cv2.solvePnP(obj_points, corners, self.camera_matrix, self.dist_coeffs)

                if retval:
                    distance_tvec = np.linalg.norm(tvec) * 1000

                    rot_matrix, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)
                    singular = sy < 1e-6
                    if not singular:
                        yaw = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
                        pitch = np.arctan2(-rot_matrix[2, 0], sy)
                        roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
                    else:
                        yaw = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
                        pitch = np.arctan2(-rot_matrix[2, 0], sy)
                        roll = 0

                    yaw, pitch, roll = np.degrees([yaw, pitch, roll])

                    msg = Aruco()
                    msg.active = True
                    msg.tag_id = int(markerID)
                    msg.distance = float(distance_tvec)
                    msg.pixel_based_distance = float(distance_pixel_method)
                    msg.tvec = tvec.flatten().tolist()
                    msg.rvec = rvec.flatten().tolist()
                    msg.corners = corners.flatten().tolist()
                    msg.yaw = float(yaw)
                    msg.pitch = float(pitch)
                    msg.roll = float(roll)

                    self.aruco_publisher.publish(msg)
                    self.get_logger().info(f'Published tag {markerID}')

                    for i in range(4):
                        pt1 = tuple(map(int, corners[i]))
                        pt2 = tuple(map(int, corners[(i + 1) % 4]))
                        cv2.line(self.video_feed, pt1, pt2, (0, 255, 0), 2)

                    cv2.drawFrameAxes(self.video_feed, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                    topLeft = tuple(map(int, corners[0]))
                    bottomRight = tuple(map(int, corners[2]))
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(self.video_feed, (cX, cY), 4, (0, 0, 255), -1)

                    cv2.putText(self.video_feed, f"ID: {markerID}", (topLeft[0], topLeft[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    text = f'Dist: {distance_tvec:.1f} mm, Pixel Dist: {distance_pixel_method:.1f} mm'
                    text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
                    cv2.putText(self.video_feed, text, (topLeft[0], topLeft[1] - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(self.video_feed, text2, (topLeft[0], topLeft[1] - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Frame", self.video_feed)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return           
        


def main(args=None):
    rclpy.init(args=args)
    aruco_node = ArucoNode()

    rclpy.spin(aruco_node)
    aruco_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
