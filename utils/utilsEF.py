import numpy as np
import igl
import scipy.sparse
import scipy.sparse.linalg as linalg
import pyshell


def computeEVLaplace(v, f, k, nonzero=True):
    """
    Compute first k eigenfunctions of area weighted laplacian

        returns:
        vals : eigenvalues
        vecs: eigenvectors n x k
    """
    k = k + 1
    l = -igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    n = np.shape(v)[0]
    vals, vecs = linalg.eigsh(l, k, m, sigma=0, which="LM")
    # remove eigenfunction corresponding to zero eigenvalue
    if nonzero:
        vals = vals[1:]
        vecs = vecs[:, 1:]

    # fix signs
    ind = vecs[0, :] < 0
    vals[ind] *= -1
    vecs[:, ind] *= -1

    return vals, vecs


def computeEV(v, f, k, bending_weight=1e-4, sigma=None, which=None, nonzero=True):
    """
    Compute eigenvectors of shell hessian 
        Args:
            fix (optional): vertex indices for boundary values

        returns:
            vals : eigenvalues
            vecs: eigenvectors 3n x k 
    """

    f = f.astype(np.int32)
    # edge flaps
    uE, EMAP, EF, EI = igl.edge_flaps(f)
    # massmatrix
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    M = scipy.sparse.block_diag((M, M, M), format='lil')
    k = k + 6

    hess = pyshell.shell_deformed_hessian(v, v, f, uE, EMAP, EF, EI, bending_weight)

    if sigma is None:
        if which is None:
            sigma = 0
            which = 'LM'
            vals, vecs = linalg.eigsh(hess, k, M=M, sigma=sigma, which=which)
        else:
            vals, vecs = linalg.eigsh(hess, k, M=M, which=which)
    else:
        vals, vecs = linalg.eigsh(hess, k, M=M, sigma=sigma)

    if nonzero:  # remove vecs corresponding to zero vals (six dimensional kernel due to rigid body motions)
        ind = vals > 1e-8

        if np.sum(ind) < k - 6:
            print('mesh has probably irregularities, found ' + str(k - 6 - np.sum(ind)) + 'many EF with zero EV')
            # vals,vecs =  linalg.eigsh(hess, k+np.sum(ind)-6,M=M,sigma = sigma,which = which)
            ind = vals > 1e-8

        vals = vals[6:]
        vecs = vecs[:, 6:]

    # fix signs
    ind = vecs[0, :] < 0
    vals[ind] *= -1
    vecs[:, ind] *= -1

    return vals, vecs


def projectOnNormals(v, f, function):
    """
    project a 3n function on vertex normals
        args:
            v, f: mesh
            function: n x 3
        returns:
            proj : scalar valued function on mesh
    """

    normals = igl.per_vertex_normals(v, f)

    proj = [normals[i] * np.dot(function[i], normals[i]) for i in range(normals.shape[0])]

    return proj


def projection(normals):
    """
    Create a projection matrix for projection on vertex normals
        args:
            normals: m x 3
        
        returns:
            P: projection matrix 3*m x m 
    
    """
    # normals in m x 3
    m = normals.shape[0]
    P = scipy.sparse.lil_matrix((3 * m, m), dtype='d')
    for j in range(m):
        P[j, j] = normals[j, 0]
        P[j + m, j] = normals[j, 1]
        P[j + 2 * m, j] = normals[j, 2]
    return P
