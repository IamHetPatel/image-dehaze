import numpy as np
import cv2 
 
import filter_bank
import weighting
import OTF
import convulation_Filter   

def CalTransmission(HazeImg, Transmission, sigma, regularize_lambda):
    # Initialize dimensions
    rows, cols = Transmission.shape

    # Load Kirsch filters
    KirschFilters = filter_bank.LoadFilterBank()

    # Normalize filters
    for idx, currentFilter in enumerate(KirschFilters):
        KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

    # Calculate weighting functions
    WFun = []
    for idx, currentFilter in enumerate(KirschFilters):
        WFun.append(weighting.CalculateWeightingFunction(HazeImg, currentFilter, sigma))

    # Precompute constants
    tF = np.fft.fft2(Transmission)
    DS = 0
    for i in range(len(KirschFilters)):
        D = OTF.psf2otf(KirschFilters[i], (rows, cols))
        DS = DS + (abs(D) ** 2)

    # Cyclic loop for refining t and u
    beta = 1
    beta_max = 2 ** 4
    beta_rate = 2 * np.sqrt(2)

    while (beta < beta_max):
        gamma = regularize_lambda / beta

        # Fix t, solve for u
        DU = 0
        for i in range(len(KirschFilters)):
            dt = convulation_Filter.circularConvFilt(Transmission, KirschFilters[i])
            u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters) * beta))), 0) * np.sign(dt)
            DU = DU + np.fft.fft2(convulation_Filter.circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

        # Fix u, solve for t
        Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
        beta = beta * beta_rate
    
    return Transmission

        
    