import numpy as np
import zero_pad


def psf2otf(psf, shape):
    
    '''
    Convert PSF (Point Spread Function) to OTF (Optical Transfer Function).
    
    Parameters:
        psf (ndarray): The input PSF (Point Spread Function).
        shape (tuple): The desired shape of the output OTF (Optical Transfer Function).
        
    Returns:
        ndarray: The computed OTF (Optical Transfer Function).
        
    Notes:
        - The PSF is first padded to the specified output size.
        - The PSF is then circularly shifted for FFT shift.
        - The FFT (Fast Fourier Transform) is computed to obtain the OTF.
        - The imaginary part of the PSF is discarded and real values are ensured to be close.
        - If the input PSF is all zeros, a zero-filled array is returned.
    '''
    
    # If the PSF is all zeros, return a zero-filled array
    if np.all(psf == 0):
        return np.zeros_like(psf)

    # Get the input shape of the PSF
    inshape = psf.shape

    # Pad the PSF to the specified output size
    psf = zero_pad.zero_pad(psf, shape, position='corner')

    # Circularly shift the PSF for FFT shift
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the FFT (Fast Fourier Transform) to obtain the OTF
    otf = np.fft.fft2(psf)

    # Discard the imaginary part of the PSF
    # Calculate the number of operations for real_if_close tolerance
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    # Discard the imaginary part and ensure real values are close
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
