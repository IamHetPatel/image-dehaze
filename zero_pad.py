import numpy as np

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    # Ensure that shape and imshape are numpy arrays of integers
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    # If the input image is already of the desired shape, return it as is
    if np.alltrue(imshape == shape):
        return image

    # Check if any dimension of the desired shape is non-positive
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    # Calculate the difference in shape between the desired and input images
    dshape = shape - imshape

    # Check if any dimension of the desired shape is smaller than the input
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    # Create a zero-filled array of the desired shape
    pad_img = np.zeros(shape, dtype=image.dtype)

    # Create indices for the input image
    idx, idy = np.indices(imshape)

    # Calculate the offset based on the specified position
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    # Copy the input image to the appropriate position in the zero-filled array
    pad_img[idx + offx, idy + offy] = image

    return pad_img
