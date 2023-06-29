import numpy as np
import utils
import scipy.sparse
import scipy.spatial
from correspondence import Shape
from tqdm import tqdm  # progress bar


class CorrespondenceSolver():
    def __init__(self, shapeA, shapeB, kmin=10, kmax=200, bending_weight=1e-4,
                 elasticBasis=False, LB=False, approx=False):
        """
        Class for computing a matching between shapeA and shapeB. Functional map C maps from B to A. Vertex Correspondence P maps from A to B. 
        See function computeCorrespondence
        
        args:
            shapeA: source shape
            shapeB: target shape
            optional:
                kmin, kmax : dimension of initial and final functional map (only quadratic functional map)
                bending_weight: weight of bending energy for computing eigenfunctions (recommend 1e-4 for shape with unit area)
                elasticBasis (boolean): uses the elastic eigenfunctions projected on normal components
                LB (boolean): uses the LB eigenfunctions
                approx (boolean): compute approximate nearest neighbor in map conversions
        """

        self.approx = approx
        self.elasticBasis = elasticBasis
        self.LBBasis = LB

        self.shapeA = shapeA
        self.shapeB = shapeB
        self.massA = shapeA.mass
        self.massB = shapeB.mass

        # precompute values
        self.invmassA = scipy.sparse.diags(1 / self.massA.diagonal())
        self.invmassB = scipy.sparse.diags(1 / self.massB.diagonal())
        self.sqrtmassA = scipy.sparse.diags(np.sqrt(self.massA.diagonal()))
        self.sqrtmassB = scipy.sparse.diags(np.sqrt(self.massB.diagonal()))

        self.bending_weight = bending_weight

        # set basisA and basisB dependent on chosen option, compute basis if not already stored in shape object
        if not self.LBBasis and not self.elasticBasis:
            print("Choice of basis not specified: elastic Basis is used")
            self.elasticBasis = True

        if self.LBBasis:
            self.basisString = 'LB basis'
            if shapeA.LBbasis is None:
                shapeA.computeLBBasis(k=kmax)
            self.basisA = shapeA.LBbasis
            if self.basisA.shape[1] < kmax:
                print('Not enough precomputed basis vectors for kmax. Basis is recomputed')
                shapeA.computeLBBasis(k=kmax)
            self.basisA = shapeA.LBbasis
            if shapeB.LBbasis is None:
                shapeB.computeLBBasis(k=kmax)
            self.basisB = shapeB.LBbasis
            if self.basisB.shape[1] < kmax:
                print('Not enough precomputed basis vectors for kmax. Basis is recomputed')
                shapeB.computeLBBasis(k=kmax)
            self.basisB = shapeB.LBbasis

        if self.elasticBasis:
            self.basisString = 'elastic basis'

            if shapeA.elasticBasis is None:
                shapeA.computeElasticBasis(k=kmax, bending_weight=bending_weight)
            self.basisA = shapeA.elasticBasis
            if self.basisA.shape[1] < kmax:
                print('Not enough precomputed basis vectors for kmax. Basis is recomputed')
                shapeA.computeElasticBasis(k=kmax, bending_weight=bending_weight)
            self.basisA = shapeA.elasticBasis
            if shapeB.elasticBasis is None:
                shapeB.computeElasticBasis(k=kmax, bending_weight=bending_weight)
            self.basisB = shapeB.elasticBasis
            if self.basisB.shape[1] < kmax:
                print('Not enough precomputed basis vectors for kmax. Basis is recomputed')
                shapeB.computeElasticBasis(k=kmax, bending_weight=bending_weight)
            self.basisB = shapeB.elasticBasis

        self.kmax = kmax
        self.kmin = kmin

    def CinitfromLandmarks(self, landmarksA, landmarksB):
        """
            Computes an inital functional map by naive alignment of first kmin basivectors on given landmark vertices.
            args:
                landmarksA, landmarksB: landmark vertices landmarksA[i] should correspond to landmarksB[i]
            
            returns:
            C_init (kminxkmin)
            
        """
        F = self.basisA[landmarksA, :self.kmin]
        G = self.basisB[landmarksB, :self.kmin]

        if self.LBBasis:
            # scalar product in subspace is euclidean
            sqrtA = np.identity(self.kmin)
        else:
            A = self.basisA[:, :self.kmin].T.dot(self.massA.dot(self.basisA[:, :self.kmin]))
            sqrtA = scipy.linalg.sqrtm(A)

        C_init = np.linalg.lstsq(np.kron(G, sqrtA), np.reshape(sqrtA.dot(F.T), (-1), order='F'), rcond=None)[0]
        C_init = np.reshape(C_init, (self.kmin, self.kmin), order='F')

        return C_init

    def funcMapfromP2P(self, P, k):
        """
            Computes a functional map (B -> A) from a v2v correspondence (A->B)  using k basis functions.
            args:
                P: v2v correspondence from A to B given as sparse matrix P in shapeB.nVert x shapeA.nVert
                k: (int) number of basis elements

            returns:
            C: (kxk) functional map approximating P
        
        """

        if self.LBBasis:
            # take advantage that basisA, basisB are orthonormal w.r.t weighted scalar product
            C = self.basisA[:, :k].T.dot(self.massA.dot(P.T.dot(self.basisB[:, :k])))
        else:
            C = np.linalg.pinv(self.sqrtmassA.dot(self.basisA[:, :k])).dot(
                self.sqrtmassA.dot(P.T.dot(self.basisB[:, :k])))

        return C

    def p2pfromFuncMap(self, C):
        """
            Estimates a v2v map from A to B for given fuctional map C (B -> A) in kxk 
            args:
                C: functional map
            returns 
                P: (shapeB.nVert x shapeA.nVert) sparse matrix encodes v2v map
        """
        P = scipy.sparse.lil_matrix((self.shapeB.nVert, self.shapeA.nVert))
        k = C.shape[0]

        if self.elasticBasis:
            # scalar product in subspaces
            A = self.basisA[:, :k].T.dot(self.massA.dot(self.basisA[:, :k]))
            B = self.basisB[:, :k].T.dot(self.massB.dot(self.basisB[:, :k]))
            sqrtA = scipy.linalg.sqrtm(A)

            dataB = sqrtA.dot(C.dot(np.linalg.inv(B).dot(self.basisB[:, :k].T))).T
            dataA = np.linalg.inv(sqrtA).dot(self.basisA[:, :k].T).T

        if self.LBBasis:
            # take advantage that basisA, basisB are orthonormal w.r.t weighted scalar product
            dataA = self.basisA[:, :k]
            dataB = self.basisB[:, :k].dot(C.T)

        dataTree = scipy.spatial.KDTree(dataB)  # input data num x dim

        if self.approx == True:
            # search for approximate nearest neighbor
            dist, ind = dataTree.query(dataA, eps=0.01, workers=6)
        else:
            # compute the nearest neighbor in dataB for a given point in dataA
            dist, ind = dataTree.query(dataA, workers=6)

        P[ind, np.arange(self.shapeA.nVert)] = 1

        return P

    def preciseMap(self, C, precompute_dmin=True):
        """
            Converts functional map C to a vertex-to-point map
            args:
                C: functional map
                precompute_dmin: (boolean) faster but heavier in memory
            
            returns:
                P_prec : sparse matrix (shapeA.nVert x shapeB.nVert), P_prec.dot(shapeB.v) gives target points of shapeA.v on surface shapeB
        """
        print("Computing vertex-to-point map with " + self.basisString)
        if self.elasticBasis:
            P_prec = utils.precise_map(self.shapeA, self.shapeB, self.basisA, self.basisB, C, orth=False,
                                       precompute_dmin=precompute_dmin)
        if self.LBBasis:
            P_prec = utils.precise_map(self.shapeA, self.shapeB, self.basisA, self.basisB, C, orth=True,
                                       precompute_dmin=precompute_dmin)
        return P_prec

    def computeCorrespondence(self, P=None, C=None, step=1):
        """
            Computes correpondence between self.shapeA and self.shapeB
            args:
                P : (numVerticesB x numVertiesA) initial p2p map A->B
                C : (kmin x kmin) initial functional map F(B) -> F(A)
                step: increase of basis functions per iteration
            returns;

        """
        print("Iterative procedure from k=" + str(self.kmin) + ' to k=' + str(self.kmax) + ' with ' + self.basisString)

        if P is None:
            if C is None:
                print("no init is given, an identity functional map will be used")
                C = np.identity(self.kmin)

            P = self.p2pfromFuncMap(C)
            self.kmin += 1

        for k in tqdm(np.arange(self.kmin, self.kmax + 1, step)):
            C = self.funcMapfromP2P(P, k)
            P = self.p2pfromFuncMap(C)

        return P, C
