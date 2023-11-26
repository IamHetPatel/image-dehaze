import cv2
import numpy as np

def AirlightEstimation(HazeImg, airlightEstimation_windowSze, A):
    """
    Estimate the airlight value for each channel in the input hazy image.

    Parameters
    ----------
    HazeImg : numpy.ndarray
        Hazy input image.
    airlightEstimation_windowSze : int
        Window size for the erosion operation.
    A : list
        Airlight values for each channel.

    Returns
    -------
    A : list
        Updated airlight values.
    """

    # Check if HazeImg is color
    if (len(HazeImg.shape) == 3):
        for ch in range(len(HazeImg.shape)):
            # Apply erosion operation to estimate the minimum intensity in the window
            kernel = np.ones((airlightEstimation_windowSze, airlightEstimation_windowSze), np.uint8)
            minImg = cv2.erode(HazeImg[:, :, ch], kernel)
            A.append(int(minImg.max()))
    else:
        # Apply erosion operation to estimate the minimum intensity in the window for grayscale image
        kernel = np.ones((airlightEstimation_windowSze, airlightEstimation_windowSze), np.uint8)
        minImg = cv2.erode(HazeImg, kernel)
        A.append(int(minImg.max()))

    return A
