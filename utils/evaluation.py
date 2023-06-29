import numpy as np
import igl
import scipy


def compute_percentageError(dist, maxdist=None, step=0.01):
    """ 
        Helper function to create geodesic error plots
        Computes percentage of values in dist below a values in np.arange(0,maxdist, step)
        dist: array of geodesic errors
        maxdist: value for largest threshold, if None max(dist) is used 
        returns 
            error: threshholds for perc computation
            perc: vector with percentages of dist below corrsponding error threshold in error
    """
    if maxdist is None:
        maxdist = max(dist)

    num = dist.size
    error = np.arange(0, maxdist, step)
    perc = np.zeros(error.size)
    for i, e in enumerate(error):
        perc[i] = sum(dist < e) / num

    return error, perc


def geodesic_distmat_dijkstra(vertices, faces, indices=None):
    """
        Compute geodesic distance matrix using Dijkstra algorithm.
        Function from pyFM 
        MIT License

        Copyright (c) 2020 Robin Magnet

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the Software), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, andor sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
            """
    print("Computing geodesic distance matrix for error evaluation.")
    N = vertices.shape[0]
    edges = igl.edges(faces)

    I = edges[:, 0]  # (p,)
    J = edges[:, 1]  # (p,)
    V = np.linalg.norm(vertices[J] - vertices[I], axis=1)  # (p,)

    In = np.concatenate([I, J])
    Jn = np.concatenate([J, I])
    Vn = np.concatenate([V, V])

    graph = scipy.sparse.coo_matrix((Vn, (In, Jn)), shape=(N, N)).tocsc()

    geod_dist = scipy.sparse.csgraph.dijkstra(graph, indices=indices)

    return geod_dist
