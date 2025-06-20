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
from interface.msg import Aruco  # Make sure this matches your actual msg path

rclpy.init()
node = rclpy.create_node('aruco_publisher')
publisher = node.create_publisher(Aruco, 'aruco_topic', 10)

package_path = get_package_share_directory('aruco_vision')
npz_path = os.path.join(package_path, 'config', 'camera_calibration_parameters.npz')

with np.load(npz_path) as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

tag_size = 0.1  # meters
focal_length = camera_matrix[0, 0]

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
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    if not ret:
        print("failed to get frame")
        break

    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(frame)

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

            retval, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

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

                publisher.publish(msg)
                node.get_logger().info(f'Published tag {markerID}')

                for i in range(4):
                    pt1 = tuple(map(int, corners[i]))
                    pt2 = tuple(map(int, corners[(i + 1) % 4]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                topLeft = tuple(map(int, corners[0]))
                bottomRight = tuple(map(int, corners[2]))
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                cv2.putText(frame, f"ID: {markerID}", (topLeft[0], topLeft[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                text = f'Dist: {distance_tvec:.1f} mm, Pixel Dist: {distance_pixel_method:.1f} mm'
                text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
                cv2.putText(frame, text, (topLeft[0], topLeft[1] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, text2, (topLeft[0], topLeft[1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
node.destroy_node()
rclpy.shutdown()
