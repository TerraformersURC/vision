import cv2
import os

# Define the directory to save calibration images
calibration_dir = 'calibration_images'
if not os.path.exists(calibration_dir):
    os.makedirs(calibration_dir)

# Initialize the camera
camera_index = 1  # Adjust the camera index if necessary
cap = cv2.VideoCapture(camera_index)

# Set camera properties (if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height

print("Press the 'space' key to capture an image.")
print("Press 'q' to exit.")

# Capture loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space key to capture an image
        # Generate filename based on the number of images already taken
        image_count = len(os.listdir(calibration_dir))
        image_path = os.path.join(calibration_dir, f'calibration_image_{image_count + 1}.jpg')
        
        # Save the captured frame
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}")
    elif key == ord('q'):  # Press 'q' to quit
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
