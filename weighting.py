import numpy as np
import convulation_Filter 

def CalculateWeightingFunction(HazeImg, Filter, sigma):
    # Calculate weighting function...
    HazeImageDouble = HazeImg.astype(float) / 255.0

    if (len(HazeImg.shape) == 3):
        # Handle RGB image
        Red = HazeImageDouble[:, :, 2]
        d_r = convulation_Filter.circularConvFilt(Red, Filter)

        Green = HazeImageDouble[:, :, 1]
        d_g = convulation_Filter.circularConvFilt(Green, Filter)

        Blue = HazeImageDouble[:, :, 0]
        d_b = convulation_Filter.circularConvFilt(Blue, Filter)
           
        return (np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * sigma * sigma)))
    else:
        # Handle grayscale image
        d = convulation_Filter.circularConvFilt(HazeImageDouble, Filter)
        return (np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma)))