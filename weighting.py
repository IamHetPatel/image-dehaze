import numpy as np
import convulation_Filter 

def CalculateWeightingFunction(HazeImg, Filter, sigma):
    """
    Calculate the weighting function based on the circular convolution of the input image with a given filter.

    Parameters
    ----------
    HazeImg : numpy.ndarray
        Input hazy image.
    Filter : numpy.ndarray
        Filter for circular convolution.
    sigma : float
        Standard deviation for the Gaussian function.

    Returns
    -------
    WeightingFunction : numpy.ndarray
        Calculated weighting function.
    """

    # Convert HazeImg to double precision in the range [0, 1]
    HazeImageDouble = HazeImg.astype(float) / 255.0

    if (len(HazeImg.shape) == 3):
        # Handle RGB image
        Red = HazeImageDouble[:, :, 2]
        d_r = convulation_Filter.circularConvFilt(Red, Filter)

        Green = HazeImageDouble[:, :, 1]
        d_g = convulation_Filter.circularConvFilt(Green, Filter)

        Blue = HazeImageDouble[:, :, 0]
        d_b = convulation_Filter.circularConvFilt(Blue, Filter)

        # Calculate and return the exponential of the negative squared sum
        return np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * sigma * sigma))
    else:
        # Handle grayscale image
        d = convulation_Filter.circularConvFilt(HazeImageDouble, Filter)

        # Calculate and return the exponential of the negative squared sum
        return np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
