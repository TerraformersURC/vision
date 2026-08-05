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
from geometry_msgs.msg import PoseStamped
from rover_interface.msg import WaypointList, WaypointData, GPSData, Controller
from std_msgs.msg import String


ARUCO_TOPIC = "/new_image"
ODOM_TOPIC  = "/zed/zed_node/odom"

class OverlayNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        self.ros_cv_bridge = CvBridge()
        self.video_feed = None

        self.camera_matrix = None
        self.dist_coeffs = None

        self.pose_subscription = self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.pose_callback, 10)

        # ODOMETRY

        # Position  (metres, in odom frame)
        self.odom_x   = 0.0
        self.odom_y   = 0.0
        self.odom_z   = 0.0
        # Orientation quaternion
        self.odom_qx  = 0.0
        self.odom_qy  = 0.0
        self.odom_qz  = 0.0
        self.odom_qw  = 1.0
        # Derived yaw (radians)
        self.odom_yaw = 0.0
        self.odom_roll  = 0.0
        self.odom_pitch = 0.0
        # Linear velocity  (m/s)
        self.odom_vx  = 0.0
        self.odom_vy  = 0.0
        self.odom_vz  = 0.0
        # Angular velocity (rad/s)
        self.odom_wx  = 0.0
        self.odom_wy  = 0.0
        self.odom_wz  = 0.0

        #STORE ARUCO INFORMATION
        package_path = get_package_share_directory('image_handler')
        npz_path = os.path.join(package_path, 'config', 'camera_calibration_parameters.npz')


        with np.load(npz_path) as data:
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']

        self.aruco_publisher = self.create_publisher(Aruco, ARUCO_TOPIC, 10)
        self.display_publisher = self.create_publisher(Image, "/overlay/image", 10)

        self.current_video_topic = None
        self.camera_subscription = None

        self.create_subscription(
            String, '/active_camera_ns', self.on_camera_switch, 10
        )


        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
        ap.add_argument("--no-display", action="store_true", help="disable OpenCV display window")
        args = vars(ap.parse_known_args()[0])

        self.no_display = args["no_display"]

        

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
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        print("[INFO] starting video stream...")

        # Waypoints
        self.circle_x = 425 # Image is ~950 across
        self.circle_y = 30
        self.circle_speed = 10

        self.timer = self.create_timer(0.03,self.detect_aruco)

        self.subscription = self.create_subscription(
            WaypointList,
            'waypoints_topic',
            self.waypoint_callback,
            10
        )

        self.waypoints = []
        self.targets = []

        self.home_lat = None  # change from hardcoded value to None
        self.home_lon = None

        self.gps_subscription = self.create_subscription(
            GPSData,
            'gps_topic',
            self.gps_callback,
            10
        )

    def on_camera_switch(self, msg: String):
        new_topic = msg.data
        if new_topic == self.current_video_topic:
            return

        # Extract namespace
        ns = '/' + new_topic.split('/')[1]

        self.current_video_topic = new_topic
        if self.camera_subscription is not None:
            self.destroy_subscription(self.camera_subscription)
        self.camera_subscription = self.create_subscription(
            Image, new_topic, self.set_videofeed_callback, 10
        )

        # Switch pose topic to match active camera
        self.destroy_subscription(self.pose_subscription)
        self.pose_subscription = self.create_subscription(
            PoseStamped, f"{ns}/zed_node/pose", self.pose_callback, 10
        )

        self.get_logger().info(f'Switched to camera namespace: {ns}')


    def set_videofeed_callback(self,msg):
        try:
            self.video_feed = self.ros_cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            print("error converting camera feed to cv2 feed")
            return
        
    def waypoint_callback(self, msg: WaypointList):
        self.waypoints = msg.list_waypoints


    def gps_callback(self, msg):
        if msg.fix_quality == 0 or msg.status == 0:
            return
        
        if self.home_lat is None:
            self.home_lat = msg.latitude
            self.home_lon = msg.longitude
            self.get_logger().info(
                f"Home set: {self.home_lat:.6f}, {self.home_lon:.6f} "
                f"({msg.num_satellites} sats, HDOP {msg.hdop:.1f})"
            )


    def latlon_to_local(self, lat, lon):
        # Convert lat/lon to local ENU metres relative to home origin
        R = 6371000.0  # Earth radius in metres

        dlat = math.radians(lat - self.home_lat)
        dlon = math.radians(lon - self.home_lon)

        x = dlon * R * math.cos(math.radians(self.home_lat))  # East
        y = dlat * R                                           # North

        return x, y


    def detect_aruco(self):
        tag_size = 0.1  # meters
        focal_length = self.camera_matrix[0, 0]


        if(self.video_feed is None):
            #print("No camera feed found!")
            return
        
            
        (corners, ids, rejected) = self.arucoDetector.detectMarkers(self.video_feed)
        
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
        self.overlay_markers()


    def color_from_string(self, color: str):
        color_map = {
            "red":    (0, 0, 255),
            "green":  (0, 255, 0),
            "blue":   (255, 0, 0),
            "yellow": (0, 255, 255),
        }
        return color_map.get(color.lower(), (255, 255, 255))
    

    def overlay_markers(self):
        
        # add padding to image
        padding_top = 60
        padding_bottom = 30
        padding_sides = 20
        display_frame = cv2.copyMakeBorder(
            self.video_feed,
            padding_top, padding_bottom, padding_sides, padding_sides,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        # display xyz rpy
        y_pos = display_frame.shape[0] - 10
        x_pos = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        labels = [
            (f'X: {self.odom_x:.2f} ',  (255, 80,  80)),   # blue-ish
            (f'Y: {self.odom_y:.2f} ',  (80,  255, 80)),    # green
            (f'Z: {self.odom_z:.2f} ',  (80,  80,  255)),   # red-ish
            (f'R: {math.degrees(self.odom_roll):.1f} ',   (255, 255, 80)),   # cyan
            (f'P: {math.degrees(self.odom_pitch):.1f} ',  (255, 80,  255)),  # magenta
            (f'Y: {math.degrees(self.odom_yaw):.1f} ',    (80,  255, 255)),  # yellow
        ]

        for text, color in labels:
            cv2.putText(display_frame, text, (x_pos, y_pos), font, scale, color, thickness)
            text_width, _ = cv2.getTextSize(text, font, scale, thickness)[0]
            x_pos += text_width
                
        # Only build targets if we have GPS home and waypoints
        if self.home_lat is not None and len(self.waypoints) > 0:
            self.targets = [
                {
                    "id": i,
                    "pos": np.array([*self.latlon_to_local(wp.latitude, wp.longitude), 0.0]),
                    "color": self.color_from_string(wp.waypoint_color)
                }
                for i, wp in enumerate(self.waypoints)
            ]

        # draw locator bar
        bar_y = 30
        bar_x_left = 0
        bar_x_right = display_frame.shape[1]
        bar_center_x = bar_x_right // 2
        bar_fov = math.radians(90)  # how many degrees the full bar width represents

        # Draw the bar
        cv2.line(display_frame, (bar_x_left, bar_y), (bar_x_right, bar_y), (60, 60, 60), 2)
        # Center tick
        cv2.line(display_frame, (bar_center_x, bar_y - 6), (bar_center_x, bar_y + 6), (255, 255, 255), 1)

        camera_pos = np.array([self.odom_x, self.odom_y, self.odom_z])

        for target in self.targets:
            # Vector from camera to target in world XY plane
            delta = target["pos"] - camera_pos
            angle_to_target = math.atan2(delta[1], delta[0])  # world-frame angle

            # Relative angle: how far left/right of camera heading
            relative_angle = angle_to_target - self.odom_yaw

            # Normalize to [-pi, pi]
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

            if abs(relative_angle) <= bar_fov / 2:
                # Map angle to pixel position
                dot_x = int(bar_center_x + (relative_angle / (bar_fov / 2)) * (bar_x_right // 2))
                dot_x = max(bar_x_left + 5, min(bar_x_right - 5, dot_x))
                cv2.circle(display_frame, (dot_x, bar_y), 6, target["color"], -1)

                cv2.putText(display_frame, str(target["id"]), (dot_x - 4, bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, target["color"], 1)
            else:
                # Draw arrow at edge pointing toward the target
                edge_x = bar_x_left + 8 if relative_angle < 0 else bar_x_right - 8
                pts = np.array([[edge_x, bar_y + (30 * target["id"])], [edge_x + (8 if relative_angle < 0 else -8), bar_y - 6 + (30 * target["id"])],
                                [edge_x + (8 if relative_angle < 0 else -8), bar_y + 6 + (30 * target["id"])]], np.int32)
                cv2.fillPoly(display_frame, [pts], target["color"])

                cv2.putText(display_frame, str(target["id"]), (edge_x - 4, bar_y + (30 * target["id"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, target["color"], 1)


        if not self.no_display:
            cv2.imshow("Frame", display_frame)
            key = cv2.waitKey(1) & 0xFF

        display_msg = self.ros_cv_bridge.cv2_to_imgmsg(display_frame, encoding='bgr8')
        self.display_publisher.publish(display_msg)

    def pose_callback(self, msg):
        self.odom_x = msg.pose.position.x
        self.odom_y = msg.pose.position.y
        self.odom_z = msg.pose.position.z

        self.odom_qx = msg.pose.orientation.x
        self.odom_qy = msg.pose.orientation.y
        self.odom_qz = msg.pose.orientation.z
        self.odom_qw = msg.pose.orientation.w

        # yaw
        siny_cosp = 2.0 * (self.odom_qw * self.odom_qz + self.odom_qx * self.odom_qy)
        cosy_cosp = 1.0 - 2.0 * (self.odom_qy ** 2 + self.odom_qz ** 2)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)  
        
        # pitch
        sinp = 2.0 * (self.odom_qw * self.odom_qy - self.odom_qz * self.odom_qx)
        self.odom_pitch = math.asin(max(-1.0, min(1.0, sinp)))

        # roll
        sinr_cosp = 2.0 * (self.odom_qw * self.odom_qx + self.odom_qy * self.odom_qz)
        cosr_cosp = 1.0 - 2.0 * (self.odom_qx ** 2 + self.odom_qy ** 2)
        self.odom_roll = math.atan2(sinr_cosp, cosr_cosp) 
        


def main(args=None):
    rclpy.init(args=args)
    overlay_node = OverlayNode()

    rclpy.spin(overlay_node)
    overlay_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
