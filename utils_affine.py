########################################################################################
# Affine transformation matrices
# From
# https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/shears.py
# https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/affines.py
# and
# https://colab.research.google.com/drive/1ImBB-N6P9zlNMCBH9evHD6tjk0dzvy1_
########################################################################################

import numpy as np
import cython
from math import floor, sqrt, atan2, isclose
from scipy.linalg import polar

def composeAffine(T, R, Z, S=None):
    ''' Compose translations, rotations, zooms, [shears]  to affine
    Parameters
    ----------
    T : array-like shape (N,)
        Translations, where N is usually 3 (3D case)
    R : array-like shape (N,N)
        Rotation matrix where N is usually 3 (3D case)
    Z : array-like shape (N,)
        Zooms, where N is usually 3 (3D case)
    S : array-like, shape (P,), optional
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine (3D case)
    Returns
    -------
    A : array, shape (N+1, N+1)
        Affine transformation matrix where N usually == 3
        (3D case)
    '''
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n,n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n,n))
    A = np.eye(n+1)
    ZS = np.diag(Z)
    if not S is None:
        ZS = ZS.dot(striu2mat(S))
    A[:n,:n] = np.dot(R, ZS)
    A[:n,n] = T[:]
    return A

def decomposeAffine23(M):
    # https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/affines.py
    # Decompose A to T,R,Z,S
    # T = [l,t,z] translation vecotr
    # R = [3x3] rotation matrix
    # Z = (2,) scale vector
    # S = (2,) shear vector sxy, syx
    # S = [1,S[0],S[1]],[0,1,S[2]],[0,0,1]
    # A = T*R*Z*S
    #
    # 2D Affine Transformations
    # -------------------------
    # With transformation matrices:
    # R = [ [cos(θ), -sin(θ), 0]
    #       [sin(θ),  cos(θ), 0]
    #       [0     ,  0     , 1] ]
    #
    # Z = [ [sx, 0, 0],  # zoom or scale
    #       [0, sy, 0],
    #       [0,  0, 1]]
    #
    # H = [ [ 1, hx, 0],  # shear
    #       [hy,  1, 0],
    #       [ 0,  0, 1]]
    #
    # T = [ [1, 0, dx],  # translation
    #       [0, 1, dy],
    #       [0, 0, 1]]
    #
    # Scaling * Shear * Translation
    # M = [ [sx,    sx*hx, sx*dx+sx*hx*dy]
    #       [sy*hy, sy,    sy*dy+sy*hy*dx]
    #       [0,     0,     1] ]
    #
    #
    # Rotation * Scaling * Shear * Translation
    # M = [ [cos(θ)*sx-sin(θ)*sy*hy,  cos(θ)*sx*hx-sin(θ)*sy, cos(θ)*(sx*dx+sx*hx*dy) - sin(θ)*(sy*dy+sy*hy*dx)]
    #       [sin(θ)*sx+cos(θ)*sy*hy,  sin(θ)*sx*hx+cos(θ)*sy, sin(θ)*(sx*dx+sx*hx*dy) + cos(θ)*(sy*dy+sy*hy*dx)]
    #       [0,                       0,                      1] ]
    #
    # Simplifications:
    # if proportional scaling sx = sy = s
    # if no shear hx = hy = 0
    # then s = sqrt(  M[0,0]*M[0,0] + M[1,0]*M[1,0] )
    # then θ = atan2( M[1,0],M[0,0] )
    # then M[0,2]/s = cos(θ)*dx - sin(θ)*dy
    # then M[1,2]/s = sin(θ)*dx + cos(θ)*dy
    
    if isclose(M[1,0], -M[0,1]): # no shear and proportional scaling
        sx  = \
        sy  = sqrt(  M[0,0]*M[0,0] + M[1,0]*M[1,0] )
        r   = atan2( M[1,0],M[0,0] )
        sxy = 1.0
        return M[:,-1], r, np.array([sx, sy]), np.array([sxy])
        
    else: # we have shear and disparate scaling
        # Prepare Matrices
        A23 = np.asarray(M)
        
        T   = A23[:,-1]
        RZS = A23[:,:-1]
        # compute scales and shears
        M0, M1 = np.array(RZS).T
        # extract x scale and normalize
        sx = sqrt(np.sum(M0**2))
        M0 /= sx
        # orthogonalize M1 with respect to M0
        sx_sxy = np.dot(M0, M1)
        M1 -= sx_sxy * M0
        # extract y scale and normalize
        sy = sqrt(np.sum(M1**2))
        M1 /= sy
        sxy = sx_sxy / sx
        # Reconstruct rotation matrix, ensure positive determinant
        R = np.array([M0, M1]).T
        if np.linalg.det(R) < 0:
            sx *= -1
            R[:,0] *= -1
        return T, R, np.array([sx, sy]), np.array([sxy])

def decomposeAffine44(M):
    # https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/affines.py
    # Decompose A to T,R,Z,S
    # T = [dx,dy,dz] translation vector
    # R = [3x3] rotation matrix
    # Z = (3,) scale vector
    # S = (3,) shear vector, the shear matrix would be [1,S[0],S[1]],[0,1,S[2]],[0,0,1]
    # A = T*R*Z*S
    
    A44 = np.asarray(M)
    T = A44[:-1,-1]
    RZS = A44[:-1,:-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize
    sy = sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= (sx_sxz * M0 + sy_syz * M1)
    # extract z scale and normalize
    sz = sqrt(np.sum(M2*M2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    R = np.array([M0, M1, M2]).T
    if np.linalg.det(R) < 0:
        sx *= -1
        R[:,0] *= -1
    return T, R, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])

def decomposeAffineCheck(M):
    # https://colab.research.google.com/drive/1ImBB-N6P9zlNMCBH9evHD6tjk0dzvy1_
    m,n = M.shape
    eye = np.eye(n)
    # Prepare Matrices
    T = eye.copy()
    H = eye.copy()
    H[0:m,0:n] = M

    # Decompose into Translation        
    T[0:m,-1] = M[0:m,-1]
    L = H.copy()
    L[:m,n-1] = 0.
    # Check that Translation is correct
    assert np.allclose(H, T @ L), 'Translation is not correct'

    # Decompose Rotation and Scaling        
    R, K = polar(L)
    if np.linalg.det(R) < 0:
        R[:m,:m] = -R[:m,:m]
        K[:m,:m] = -K[:m,:m]
    # Check that R,K composition is correct
    assert np.allclose(L, R @ K), 'Rotation and Scaling is not correct'
    assert np.allclose(H, T @ R @ K), 'Rotation and Scaling is not correct'
    
    # Decompose Scaling
    Z=[]
    f, X = np.linalg.eig(K)             # f are the scales, X are the axes
    for factor, axis in zip(f, X.T):
        if not np.isclose(factor, 1):
            scale = np.eye(n) + np.outer(axis, axis) * (factor-1)
            Z.append(scale)        
    # Check that scale composition is correct
    Z_check = eye.copy()
    for scale in Z:
        Z_check = Z_check @ scale            
    assert np.allclose(K, Z_check), 'Scaling is not correct'        
    # Check that completed composition is correct
    assert np.allclose(H, T @ R @ Z_check), 'Decomposition is not correct'
        
    return T, R, Z

# Caching dictionary for common shear Ns, indices
_shearers = {}
for n in range(1,11):
    x = (n**2 + n)/2.0
    i = n+1
    _shearers[x] = (i, np.triu(np.ones((i,i)), 1).astype(bool))
    
def striu2mat(striu):
    ''' Construct shear matrix from upper triangular vector
    Parameters
    ----------
    striu : array, shape (N,)
       vector giving triangle above diagonal of shear matrix.
    Returns
    -------
    SM : array, shape (N, N)
       shear matrix
    Examples
    --------
    >>> S = [0.1, 0.2, 0.3]
    >>> striu2mat(S)
    array([[1. , 0.1, 0.2],
           [0. , 1. , 0.3],
           [0. , 0. , 1. ]])
    >>> striu2mat([1])
    array([[1., 1.],
           [0., 1.]])
    >>> striu2mat([1, 2])
    Traceback (most recent call last):
       ...
    ValueError: 2 is a strange number of shear elements
    Notes
    -----
    Shear lengths are triangular numbers.
    See http://en.wikipedia.org/wiki/Triangular_number
    '''
    # https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/shears.py
    # https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/affines.py

    n = len(striu)
    # cached case
    if n in _shearers:
        N, inds = _shearers[n]
    else: # General case
        N = ((-1+sqrt(8*n+1))/2.0)+1 # n+1 th root
        if N != floor(N):
            raise ValueError('%d is a strange number of shear elements' %
                             n)
        N = int(N)
        inds = np.triu(np.ones((N,N)), 1).astype(bool)
    M = np.eye(N)
    M[inds] = striu
    return M

########################################################################################
# Algin two point clouds
########################################################################################

def align(src, dst, estimate_scale=True):
    """
    Shinji Umeyama, 
    Least-Squares Estimation of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
    DOI: 10.1109/34.88573
    Input: (works with 2D, 3D and nD point pairs)
        src -- source points (M=num_points,N=dimensions) e.g.(5,2)
        dst -- matching destination points (M,N)
        estimate_scale -- wether to estimate scaling factor
    Output:
    T   -- the homogenous similarity transformation matrix, (N+1,N+1) e.g.(3,3)
           contains NaN if not well conditioned
    """
    # From:
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py
    # https://github.com/uzh-rpg/rpg_vikit
    # https://gist.github.com/CarloNicolini/7118015
    # Example application:
    # https://github.com/shaoanlu/faceswap-GAN/blob/master/umeyama.py
    # https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_py/src/vikit_py/align_trajectory.py

    src = np.asarray(src)
    dst = np.asarray(dst)

    (num, dim) = src.shape

    # mean of src and dst
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # substract mean
    src_zeromean = src - src_mean
    dst_zeromean = dst - dst_mean

    # correlation, Eq. 38
    A = np.matmul(dst_zeromean.T, src_zeromean) / num
        
    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0: 
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    # Singular Value Decomposition
    # A = U*S*Vh with 
    # diag(S) the eigenvalues
    # Vh the eigenvectors of (Ah * A)
    # U the eigenvectors of (A*Ah)

    U,S,V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.matmul(U, V) # U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.matmul(U, np.matmul(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.matmul(U, np.matmul(np.diag(d), V))

    if estimate_scale:
        scale = 1./src_zeromean.var(axis=0).sum() * np.matmul(S, d)
    else:
        scale = 1.

    # translation column
    T[:dim, dim]   = dst_mean - scale*(np.matmul(T[:dim, :dim], src_mean.T))
    #                (np.matmul(T[:dim, :dim], dst_mean.T)/scale - src_mean)*scale
    # rotation, scale
    T[:dim, :dim] *= scale
            
    return T
