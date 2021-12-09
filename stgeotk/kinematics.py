import concurrent.futures
import math
import multiprocessing
import scipy.linalg as LA
import numpy as np
from . utility import logger


def _strain_ellipsoid_single_entry(F):
    '''
    Given the deformation gradient F,
    extract lineation, foliation normal
    and the intermediate strain axis
    which correspond to the 3 axis of the strain ellipsoid.
    They are also the eigenvectors of B = F*F^{T} = V*V,
    which is the left Cauchy-Green Tensor.
    Return the values as 3x3 matrix, the row from top to bottom are
    foliation normal, intermediate axis and lineation
    '''
    try:
        FFt = F.dot(F.T)
        eigval, eigvec = LA.eigh(FFt)
        for i in range(0, 3):
            eigvec[:, i] *= math.sqrt(eigval[i])
    except ValueError:
        # make note of the matrix
        logger.exception(
            "The matrix is ill-conditioned and numerical algorithms have broken down.\n")
        logger.error("%s", "The F matrix is:\n{0}\n".format(F))
        logger.error("%s", "The FFt matrix is:\n{0}\n".format(FFt))
        logger.error("%s", "Eigenvalues are: {0}\n".format(eigval))
        raise  # rethrow the error
    return eigvec.T


def _strain_ellipsoid_multiprocess(F, range_begin, range_end):
    eigvecs = np.zeros((range_end-range_begin, 3, 3))
    counter = 0
    for i in range(range_begin, range_end):
        eigvecs[counter, :] = _strain_ellipsoid_single_entry(F[i, :])
        counter += 1
    return eigvecs


def strain_ellipsoid(F):
    if len(F.shape) == 2 and F.shape == (3, 3):
        return _strain_ellipsoid_single_entry(F)
    elif len(F.shape) == 3 and F.shape[1] == F.shape[2] == 3:
        # use multi-processing
        n_procs = multiprocessing.cpu_count()
        idx = (F.shape[0] + np.arange(0, n_procs))//n_procs
        idx = np.concatenate(([0], np.cumsum(idx)))
        eigvecs = np.empty((0, 3, 3))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            eigvecs_chunks = [executor.submit(_strain_ellipsoid_multiprocess,
                                              F, idx[iproc], idx[iproc+1]) \
                                              for iproc in range(0, n_procs)]
            # retrieve the results in the future object
            for chunk in eigvecs_chunks:
                eigvecs = np.concatenate((eigvecs, chunk.result()), axis=0)
        return eigvecs
    else:
        raise RuntimeError("Unexpected input shape for F: {0}".format(F.shape))


def polar_decomposition(F, mode='left'):
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


def det3(A):
    '''
    Analytical implementation of the determinant of a 3x3 matrix.
    This might break down if the matrix has huge condition number.
    '''
    if __debug__:
        assert A.shape == (3, 3)
    val = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * \
        A[1, 2] * A[2, 0] + A[0, 2] * A[1, 0] * A[2, 1]
    val -= A[0, 2]*A[1, 1]*A[2, 0] + A[0, 0] * \
        A[1, 2]*A[2, 1] + A[0, 1]*A[1, 0]*A[2, 2]
    return val


def eigh3_analytical(A):
    '''
    A specialization for eigenvalue/eigenvector computation
    for 3x3 symmetric matrix
    Reference:
    Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin.
    Closed-form expressions of the eigen decomposition of
    2 x 2 and 3 x 3 Hermitian matrices.
    [Research Report] Université de Lyon. 2017.
    https://hal.archives-ouvertes.fr/hal-01501221/document
    '''
    if __debug__:
        assert A.shape == (3, 3)

    a = A[0, 0]
    b = A[1, 1]
    c = A[2, 2]
    d = A[1, 0]
    e = A[2, 1]
    f = A[2, 0]
    abc = a + b + c

    ra = 2.*a - b - c
    rb = 2.*b - a - c
    rc = 2.*c - a - b

    x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3 * (d*d + f*f + e*e)
    x2 = -ra*rb*rc + 9.0 * (rc*d*d + rb*f*f + ra*e*e) - 54.*(d*e*f)

    tmp = math.sqrt(4.0*x1*x1*x1 - x2*x2)

    if x2 > 0:
        phi = math.atan(tmp/x2)
    elif x2 < 0:
        phi = math.atan(tmp/x2) + math.pi
    else:
        phi = 0.5 * math.pi

    # eigenvalues
    lambda1 = (abc - 2.0*math.sqrt(x1)*math.cos(phi/3.0)) / 3.0
    lambda2 = (abc + 2.0*math.sqrt(x1)*math.cos((phi - math.pi)/3.0)) / 3.0
    lambda3 = (abc + 2.0*math.sqrt(x1)*math.cos((phi + math.pi)/3.0)) / 3.0

    m1 = (d*(c - lambda1) - e*f) / (f*(b-lambda1) - d*e)
    m2 = (d*(c - lambda2) - e*f) / (f*(b-lambda2) - d*e)
    m3 = (d*(c - lambda3) - e*f) / (f*(b-lambda3) - d*e)

    v1 = np.array([(lambda1 - c - e*m1)/f, m1, 1.0])
    v2 = np.array([(lambda2 - c - e*m2)/f, m2, 1.0])
    v3 = np.array([(lambda3 - c - e*m3)/f, m3, 1.0])

    v = [v1, v2, v3]
    l = [lambda1, lambda2, lambda3]

    v_sorted = []
    l_sorted = []
    for (x, y) in sorted(zip(l, v)):
        l_sorted.append(x)
        v_sorted.append(y / np.linalg.norm(y))

    return np.array(l_sorted), np.array(v_sorted).T


def kinematic_vorticity(L_tensor):
    '''
    Given the deformation gradient tensor
    $L_{ij} = frac{partial v_i}{partial x_j}$
    compute the kinematic vorticity
    The explicit formula is in taken from:
    Tikoff and Fossen, Vol 17, No 12, 1995, Journal of Structural Geology
    "The limitations of three-dimensional kinematic vorticity analysis", Eqn(4)
    '''

    # first extract entries of rate-of-deformation tensor D
    if len(L_tensor.shape) == 2:
        def L(i, j):
            return L_tensor[i-1, j-1]
    elif len(L_tensor.shape) == 3:
        # multiple entries
        def L(i, j):
            return L_tensor[:, i-1, j-1]
    else:
        raise RuntimeError(f"Unknown input dimension for L: {L_tensor.shape}")

    # norm squared of vorticity
    W2 = (L(2, 3) - L(3, 2))**2
    W2 += (L(1, 3) - L(3, 1))**2
    W2 += (L(1, 2) - L(2, 1))**2

    # squared stretching
    # in fact this is the J2 invariant of rate-of-deformation D
    S2 = 2.0 * (L(1, 1)**2 + L(2, 2)**2 + L(3, 3)**2)
    S2 += (L(1, 2) + L(2, 1))**2
    S2 += (L(2, 3) + L(3, 2))**2
    S2 += (L(1, 3) + L(3, 1))**2

    return np.sqrt(W2/S2)


def strain_rate(L):
    '''
    Compute the double contraction of
    the rate-of-deformation tensor D, i.e. D:D
    where L = D + W
    '''
    if len(L.shape) == 2:
        return np.linalg.norm(0.5 * (L + L.T))
    elif len(L.shape) == 3:  # array of L tensors
        D = 0.5 * (L + L.transpose((0, 2, 1)))
        return np.linalg.norm(D, axis=(1, 2))
    else:
        raise RuntimeError(f"Unknown input dimension for L: {L.shape}")
