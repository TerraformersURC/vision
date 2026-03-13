import pyrealsense2 as rs
import cv2 as cv
import numpy as np

FPS = 30

class RealsenseCam():
    def __init__(self) -> None:
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, *(848, 480), rs.format.z16, FPS)
        self.config.enable_stream(rs.stream.color, *(1920, 1080), rs.format.bgr8, FPS)
        self.config.enable_stream(rs.stream.infrared, 1, *(848, 480), rs.format.y8, FPS)

        self.pipeline = rs.pipeline()
        self.pipeline.start(self.config)

        self.frames = self.pipeline.wait_for_frames()
 
    def get_depth_frame(self) -> np.ndarray:
        depth_frame = self.frames.get_depth_frame()
        depth_colormap = cv.applyColorMap(
            cv.convertScaleAbs(
                np.asarray(depth_frame.get_data()),
                alpha=0.03
            ),
            cv.COLORMAP_JET
        )
        return depth_colormap

    def get_color_frame(self) -> np.ndarray:
        color_frame = self.frames.get_color_frame.get_data()
        return np.asarray(color_frame)

    def get_infrared_frame(self) -> np.ndarray:
        infrared_frame = self.frames.get_color_frame.get_data()
        return np.asarray(color_frame)


if __name__ == "__main__":
    
    cam = RealsenseCam()
    print(cam.pipeline.get_active_profile().get_streams())

    while True:
        cv.imshow('frame', cam.get_color_frame())

        if cv.waitKey(1) == ord('q'):
            break

        