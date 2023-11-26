import numpy as np
    
def LoadFilterBank():
    """ Load filter bank """
    KirschFilters = []

    # Create and append filters to the list
    KirschFilters.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
    KirschFilters.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
    KirschFilters.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]))

    return (KirschFilters)