import argparse
import time
import cv2
import sys
import math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

with np.load('camera_calibration_parameters.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Define the actual size of the AprilTag in meters
tag_size = 0.1  # Example: 10 cm

# Extract the focal length (fx) from the camera matrix
focal_length = camera_matrix[0, 0]  # fx from the intrinsic matrix

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

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
arucoParams = cv2.aruco.DetectorParameters()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
	ret, frame = vs.read()
	# detect ArUco markers in the input frame

	if not ret:
		print("failed to get frame")
		break

	detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
	(corners, ids, rejected) = detector.detectMarkers(frame)


	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			corners = markerCorner.reshape((4, 2))
			print(corners)


			tag_pixel_size = (np.linalg.norm(corners[0] - corners[1]) +
                          np.linalg.norm(corners[1] - corners[2]) +
                          np.linalg.norm(corners[2] - corners[3]) +
                          np.linalg.norm(corners[3] - corners[0])) / 4

			# Calculate the distance using the pixel size and convert to millimeters
			distance_pixel_method = (tag_size * focal_length) / tag_pixel_size * 1000

			# Define the 3D coordinates of the tag's corners in the tag's coordinate frame
			obj_points = np.array([[-tag_size / 2, tag_size / 2, 0],
								[ tag_size / 2, tag_size / 2, 0],
								[ tag_size / 2,  -tag_size / 2, 0],
								[-tag_size / 2,  -tag_size / 2, 0]], dtype=np.float32)

			# Estimate the pose of the tag
			retval, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

			if retval:
				# Draw the detected tag corners on the frame
				for i in range(4):
					pt1 = tuple(map(int, corners[i]))
					pt2 = tuple(map(int, corners[(i + 1) % 4]))
					cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

				# Convert rotation vector to rotation matrix
				rot_matrix, _ = cv2.Rodrigues(rvec)

				# Calculate yaw, pitch, and roll from the rotation matrix
				sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
				singular = sy < 1e-6
				if not singular:
					yaw = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
					pitch = np.arctan2(-rot_matrix[2, 0], sy)
					roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
				else:
					yaw = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
					pitch = np.arctan2(-rot_matrix[2, 0], sy)
					roll = 0

				# Convert angles to degrees
				yaw, pitch, roll = np.degrees([yaw, pitch, roll])

				# Calculate the distance using the translation vector and convert to millimeters
				distance_tvec = np.linalg.norm(tvec) * 1000  # Convert to mm

				# Display the tag's information on the frame
				tag_id = markerID
				text = f'ID: {tag_id}, Dist: {distance_tvec:.1f} mm, Pixel Dist: {distance_pixel_method:.1f} mm'
				text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
				cv2.putText(frame, text, (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				cv2.putText(frame, text2, (pt1[0], pt1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

				# Draw the coordinate axes on the tag
				cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

				# Print the position and distances
				print(f"Tag ID: {tag_id} - Position (x, y, z): {tvec.flatten()} - Distance (tvec): {distance_tvec:.1f} mm")
				print(f"Pixel-based Distance: {distance_pixel_method:.1f} mm")
				print(f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}")

			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
	
    # draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the frame
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()