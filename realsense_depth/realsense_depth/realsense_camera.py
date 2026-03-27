import pyrealsense2 as rs
import cv2 as cv
import numpy as np

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
        return 

if __name__ == "__main__":
    cam = RealsenseCam()

    print(cam.pipeline.get_active_profile().get_streams())

    while True:
        image = cam.get_depth_frame()
        cv.imshow('frame', image)

        if cv.waitKey(1) == ord('q'):
            break