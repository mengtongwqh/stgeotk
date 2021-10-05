import math
import scipy.linalg as LA


def strain_ellipsoid(F):
    '''
    Given the deformation gradient F, 
    extract lineation, foliation normal 
    and the intermediate strain axis
    which correspond to the 3 axis of the strain ellipsoid.
    They are also the eigenvectors of B = F*F^{T} = V*V, 
    which is the left Cauchy-Green Tensor. 
    '''
    if __debug__:
       assert F.shape == (3,3)

    # R, V  = LA.polar(F, "left")  # V*V = F*F^{T}
    eigval, eigvec = LA.eigh(F.dot(F.T))
    lineation =  math.sqrt(eigval[2]) * eigvec[:,2]
    foliation_normal = math.sqrt(eigval[0]) * eigvec[:, 0]
    intermediate_axis = math.sqrt(eigval[1]) * eigvec[:,1]
    return lineation, foliation_normal, intermediate_axis
    

def polar_decomposition(F, mode = 'left'):
    '''
    compute polar decomposition of deformation gradient tensor.
    if mode is left, 
    compute VR = F, V is the left stretch, 
    V*V is left Cauchy-Green tensor
    if mode is right, compute RU = F, U is the right stretch,
    U*U is the right Cauchy-Green tensor.
    '''
    rotation, stretch = LA.polar(F, mode)
    return stretch, rotation


def rotation_matrix_to_euler_angles(R):
    '''
    Given the rotation matrix R which is 3x3,
    compute the Euler angles on each axis.
    '''
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1.0e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return x, y, z


