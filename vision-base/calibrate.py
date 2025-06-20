import numpy as np
import cv2
import glob
import os

# Parameters
CHECKERBOARD = (10, 7)  # Specify the dimensions of the checkerboard used for calibration
square_size = 0.025  # Set the size of a square in the checkerboard (meters)

# Termination criteria for the cornerSubPix algorithm
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

# Prepare object points for a 3D grid, e.g., (0,0,0), (1,0,0), ..., (10,7,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # Scale object points to the actual size of the checkerboard squares

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane
failed_images = []  # List to store filenames of images where detection failed

# Load images for calibration
calibration_dir = 'calibration_images'  # Directory where captured images are stored
images = glob.glob(os.path.join(calibration_dir, '*.jpg'))

# Process each image to find corners
for fname in images:
    img = cv2.imread(fname)
    
    if img is None:
        print(f"Error: Unable to read {fname}")
        failed_images.append(fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners (optional)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"Warning: Checkerboard corners not found in {fname}")
        failed_images.append(fname)

cv2.destroyAllWindows()

# Perform camera calibration if enough valid images were found
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Save camera parameters to a file
    output_file = 'camera_calibration_parameters.npz'
    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
    print(f"Camera calibration parameters saved to {output_file}")

    # Check the reprojection error (optional)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    total_error = mean_error / len(objpoints)
    print(f"Total Reprojection Error: {total_error}")
    # Categorize the error
    if total_error < 0.5:
        print("Calibration Quality: Excellent")
    elif total_error < 1.0:
        print("Calibration Quality: Good")
    elif total_error < 2.0:
        print("Calibration Quality: Acceptable")
    else:
        print("Calibration Quality: Poor")
else:
    print("Error: No valid images found for calibration.")

# Save the list of failed images to a text file
failed_images_file = 'failed_images.txt'
with open(failed_images_file, 'w') as f:
    for item in failed_images:
        f.write(f"{item}\n")

if failed_images:
    print(f"Failed to process {len(failed_images)} images. Check {failed_images_file} for the list of files to retake.")
