import pyrealsense2 as rs
import cv2 as cv
import numpy as np

pipeline = rs.pipeline()
pipeline.start()

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    depth_colormap = cv.applyColorMap(
        cv.convertScaleAbs(
            np.asanyarray(depth_frame.get_data()),
            alpha=0.03
        ),
        cv.COLORMAP_JET
    )
    cv.imshow('frame', depth_colormap)

    if cv.waitKey(1) == ord('q'):
        break

pipeline.stop()