import numpy as np
import copy
import cv2

import airlight_estimation
import bound_con
import cal_Trasmission

class image_dehazer():
    def __init__(self, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                 regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
        """
        Initialize the Image Dehazer with default parameters.

        Parameters
        ----------
        airlightEstimation_windowSze : int, optional
            Window size for airlight estimation, default is 15.
        boundaryConstraint_windowSze : int, optional
            Window size for boundary constraint, default is 3.
        C0 : int, optional
            Parameter for boundary constraint, default is 20.
        C1 : int, optional
            Parameter for boundary constraint, default is 300.
        regularize_lambda : float, optional
            Regularization parameter, default is 0.1.
        sigma : float, optional
            Parameter for transmission calculation, default is 0.5.
        delta : float, optional
            Fine-tuning parameter for dehazing, default is 0.85.
        showHazeTransmissionMap : bool, optional
            Flag to show the haze transmission map, default is True.

        Attributes
        ----------
        _A : list
            Estimated airlight.
        _transmission : list
            Estimated transmission.
        _WFun : list
            Weight function.
        """
        self.airlightEstimation_windowSze = airlightEstimation_windowSze
        self.boundaryConstraint_windowSze = boundaryConstraint_windowSze
        self.C0 = C0
        self.C1 = C1
        self.regularize_lambda = regularize_lambda
        self.sigma = sigma
        self.delta = delta
        self.showHazeTransmissionMap = showHazeTransmissionMap
        self._A = []
        self._transmission = []
        self._WFun = []

    def __removeHaze(self, HazeImg):
        '''
        Remove haze from the input hazy image.

        Parameters
        ----------
        HazeImg : numpy.ndarray
            Hazy input image.

        Returns
        -------
        HazeCorrectedImage : numpy.ndarray
            Dehazed image.
        '''
        # This function will implement equation(3) in the paper
        # "https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf"

        epsilon = 0.0001
        Transmission = pow(np.maximum(abs(self._transmission), epsilon), self.delta)

        HazeCorrectedImage = copy.deepcopy(HazeImg)
        if (len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                temp = ((HazeImg[:, :, ch].astype(float) - self._A[ch]) / Transmission) + self._A[ch]
                temp = np.maximum(np.minimum(temp, 255), 0)
                HazeCorrectedImage[:, :, ch] = temp
        else:
            temp = ((HazeImg.astype(float) - self._A[0]) / Transmission) + self._A[0]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage = temp
        return HazeCorrectedImage

    def remove_haze(self, HazeImg):
        '''
        Remove haze from the input hazy image.

        Parameters
        ----------
        HazeImg : numpy.ndarray
            Hazy input image.

        Returns
        -------
        haze_corrected_img : numpy.ndarray
            Dehazed image.
        HazeTransmissionMap : numpy.ndarray
            Haze transmission map.
        '''
        A = airlight_estimation.AirlightEstimation(HazeImg, self.airlightEstimation_windowSze, self._A)
        self._A = A

        A, Transmission, C0, C1 = bound_con.BoundCon(HazeImg, self.boundaryConstraint_windowSze, self._A,
                                                     self._transmission, self.C0, self.C1)
        self._A, self._transmission, self.C0, self.C1 = A, Transmission, C0, C1

        Transmission = cal_Trasmission.CalTransmission(HazeImg, self._transmission, self.sigma, self.regularize_lambda)
        if (self.showHazeTransmissionMap):
            cv2.imshow("Haze Transmission Map", Transmission)
            cv2.waitKey(1)
        
        self._transmission = Transmission

        haze_corrected_img = self.__removeHaze(HazeImg)
        HazeTransmissionMap = self._transmission
        return haze_corrected_img, HazeTransmissionMap

def remove_haze(HazeImg, airlightEstimation_windowSze=15, boundaryConstraint_windowSze=3, C0=20, C1=300,
                regularize_lambda=0.1, sigma=0.5, delta=0.85, showHazeTransmissionMap=True):
    '''
    Remove haze from the input hazy image using the image_dehazer class.

    Parameters
    ----------
    HazeImg : numpy.ndarray
        Hazy input image.
    airlightEstimation_windowSze : int, optional
        Window size for airlight estimation, default is 15.
    boundaryConstraint_windowSze : int, optional
        Window size for boundary constraint, default is 3.
    C0 : int, optional
        Parameter for boundary constraint, default is 20.
    C1 : int, optional
        Parameter for boundary constraint, default is 300.
    regularize_lambda : float, optional
        Regularization parameter, default is 0.1.
    sigma : float, optional
        Parameter for transmission calculation, default is 0.5.
    delta : float, optional
        Fine-tuning parameter for dehazing, default is 0.85.
    showHazeTransmissionMap : bool, optional
        Flag to show the haze transmission map, default is True.

    Returns
    -------
    HazeCorrectedImg : numpy.ndarray
        Dehazed image.
    HazeTransmissionMap : numpy.ndarray
        Haze transmission map.
    '''
    Dehazer = image_dehazer(airlightEstimation_windowSze=airlightEstimation_windowSze,
                            boundaryConstraint_windowSze=boundaryConstraint_windowSze, C0=C0, C1=C1,
                            regularize_lambda=regularize_lambda, sigma=sigma, delta=delta,
                            showHazeTransmissionMap=showHazeTransmissionMap)
    HazeCorrectedImg, HazeTransmissionMap = Dehazer.remove_haze(HazeImg)
    return HazeCorrectedImg, HazeTransmissionMap

    