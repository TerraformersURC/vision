import cv2
import numpy as np
from apriltag import apriltag

# Load the camera calibration data
with np.load('camera_calibration_parameters.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Define the actual size of the AprilTag in meters
tag_size = 0.06  # Example: 10 cm

# Extract the focal length (fx) from the camera matrix
focal_length = camera_matrix[0, 0]  # fx from the intrinsic matrix

# Initialize the AprilTag detector
# Use euler angles
detector = apriltag("tagStandard41h12")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for AprilTag detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    detections = detector.detect(gray)

    # Process each detected tag
    for detection in detections:
        # Extract corner points of the detected AprilTag
        corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)

        # Calculate the pixel size of the tag using the average of all four edges
        tag_pixel_size = (np.linalg.norm(corners[0] - corners[1]) +
                          np.linalg.norm(corners[1] - corners[2]) +
                          np.linalg.norm(corners[2] - corners[3]) +
                          np.linalg.norm(corners[3] - corners[0])) / 4

        # Calculate the distance using the pixel size and convert to millimeters
        distance_pixel_method = (tag_size * focal_length) / tag_pixel_size * 1000  # Convert to mm

        # Define the 3D coordinates of the tag's corners in the tag's coordinate frame
        obj_points = np.array([[-tag_size / 2, -tag_size / 2, 0],
                               [ tag_size / 2, -tag_size / 2, 0],
                               [ tag_size / 2,  tag_size / 2, 0],
                               [-tag_size / 2,  tag_size / 2, 0]], dtype=np.float32)

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
            tag_id = detection['id']
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

    # Display the resulting frame
    cv2.imshow('AprilTag Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
