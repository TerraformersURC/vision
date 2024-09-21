import cv2
from imutils.video import Video

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)


