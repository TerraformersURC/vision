from realsense_camera import RealsenseCam
import cv2 as cv

cam = RealsenseCam()

while True:
    cv.imshow("frame", cam.get_depth_colormap())

    if cv.waitKey(1) == ord('q'):
        break