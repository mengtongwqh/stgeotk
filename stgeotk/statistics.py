import numpy as np


def mean_vector(xyz):
    """
    Compute the mean vector of a vector dataset
    return the mean vector and the spherical variance s_s
    """
    assert xyz.shape[1] == 3, "The input data must be an N x 3 array"
    n = xyz.shape[0]
    meanvector = xyz.sum(axis=0)
    R = np.linalg.norm(meanvector)
    meanvector = meanvector / R
    return meanvector, 1 - R/n


def orientation_tensor(xyz, normalize = True):
    """
    return the orientation tensor defined in:
    N. H. WOODCOCK; 
    Specification of fabric shapes using an eigenvalue method. 
    GSA Bulletin 1977; 88 (9): 1231–1236. 
    doi: https://doi.org/10.1130/0016-7606(1977)88<1231:SOFSUA>2.0.CO;2
    """
    assert xyz.shape[1] == 3, "The input data must be an N x 3 array"
    n = xyz.shape[0]
    T = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,i+1):
            T[i,j] = T[j,i] =  np.inner(xyz[:,i], xyz[:,j])
    
    return T/n if normalize else T

