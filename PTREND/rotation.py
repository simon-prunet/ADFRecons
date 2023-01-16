import numpy as np
from numba import njit
kwd = {"cache": True, "fastmath": {"reassoc", "contract", "arcp"}}

@njit(**kwd)
def rotation(angle,axis):
    '''
    Compute rotation matrix from angle and axis coordinates,
    using Rodrigues formula
    '''
    ca = np.cos(angle)
    sa = np.sin(angle)

    cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    mat = np.eye(3) + sa*cross + (1.0-ca)*np.dot(cross,cross)
    return (mat)

