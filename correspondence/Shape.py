import utils as utils
import numpy as np
import igl


class Shape():
    def __init__(self, v, f, name='', Evalues=None, elasticBasis=None, LBbasis=None, rescale_unit_area=False):
        """
        Args:
            v,f : triangulation
            name :(str) name of Shape, will be used to save results
                elasticBasis: (nVertxk)
        """
        if rescale_unit_area:
            area = np.sqrt(np.sum(igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)))
            v = (v / area)
        self.v = v
        self.f = f.astype(np.int32)
        self.name = name

        self.nVert = v.shape[0]
        self.mass = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        self.normals = igl.per_vertex_normals(self.v, self.f)
        self.adj = igl.adjacency_list(self.f)
        self.bending_weight = None
        self.vectorBasis = None
        self.elasticBasis = elasticBasis
        self.LBbasis = LBbasis
        self.Evalues = Evalues

    def computeVectorBasis(self, k=200, bending_weight=1e-4, nonzero=True):
        """
            computes elastic vibration modes (i.e. vector valued eigenfunctions of elastic hessian)
            args:
                k: first k eigennfunctions (lowest eigenvalues)
                bending_weight: weighting term of membrane and bending energy 
                nonzero: (boolean) compute eigenfunctions with nonzero eigenvalue (kernel with r.b.m. is removed)
        """
        print("Computing " + str(k) + ' elastic eigenfunctions for ' + self.name)
        self.bending_weight = bending_weight
        self.Evalues, self.vectorBasis = utils.computeEV(self.v, self.f, k, bending_weight=bending_weight,
                                                         nonzero=nonzero)

    def computeElasticBasis(self, k=200, bending_weight=1e-4, nonzero=True):
        if self.vectorBasis is None or self.vectorBasis.shape[1] < k:
            self.computeVectorBasis(k, bending_weight, nonzero=nonzero)

        self.elasticBasis = utils.projection(self.normals).T.dot(self.vectorBasis)

    def computeLBBasis(self, k=200, nonzero=True):
        print("Computing " + str(k) + ' LB eigenfunctions for ' + self.name)
        self.Evalues, self.LBbasis = utils.computeEVLaplace(self.v, self.f, k, nonzero=nonzero)
