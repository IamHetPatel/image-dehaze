import cv2
import numpy as np

def AirlightEstimation(HazeImg, airlightEstimation_windowSze, A):
    if (len(HazeImg.shape) == 3):
        for ch in range(len(HazeImg.shape)):
            kernel = np.ones((airlightEstimation_windowSze, airlightEstimation_windowSze), np.uint8)
            minImg = cv2.erode(HazeImg[:, :, ch], kernel)
            A.append(int(minImg.max()))
    else:
        kernel = np.ones((airlightEstimation_windowSze, airlightEstimation_windowSze), np.uint8)
        minImg = cv2.erode(HazeImg, kernel)
        A.append(int(minImg.max()))
    return A
    