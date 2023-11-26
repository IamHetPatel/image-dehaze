import numpy as np
import cv2

def circularConvFilt(Img, Filter):
    """
    Perform circular convolution filter operation on the input image.

    Parameters
    ----------
    Img : numpy.ndarray
        Input image.
    Filter : numpy.ndarray
        Circular convolution filter.

    Returns
    -------
    Result : numpy.ndarray
        Result of the circular convolution filter operation.
    """

    # Get the dimensions of the filter
    FilterHeight, FilterWidth = Filter.shape

    # Check if the filter is square and has odd dimensions
    assert (FilterHeight == FilterWidth), 'Filter must be square in shape.'
    assert (FilterHeight % 2 == 1), 'Filter dimension must be an odd number.'

    # Calculate half size of the filter
    filterHalsSize = int((FilterHeight - 1) / 2)

    # Get the dimensions of the input image
    rows, cols = Img.shape

    # Perform circular padding using cv2.copyMakeBorder with BORDER_WRAP
    PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize,
                                   borderType=cv2.BORDER_WRAP)

    # Perform 2D convolution using cv2.filter2D
    FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)

    # Extract the result by removing the circular padding
    Result = FilteredImg[filterHalsSize:rows + filterHalsSize, filterHalsSize:cols + filterHalsSize]

    return Result
