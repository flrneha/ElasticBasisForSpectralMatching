import numpy as np
import igl
import scipy
import polyscope as ps
import os
from correspondence import Shape
from correspondence import CorrespondenceSolver

if __name__ == "__main__":
    readPath = 'data/'
    writePath = 'results/'
    if not os.path.exists(writePath):
        os.mkdir(writePath)
    nameA = 'lion'
    nameB = 'cat'

    bending_weight = 1e-4
    kmin = 35
    kmax = 200
    precise = False  # convert the final vertex map to a vertex-to-point map

    vA, textA, n, fA, m, b = igl.read_obj(readPath + nameA + '.obj')
    vB, textB, n, fB, m, b = igl.read_obj(readPath + nameB + '.obj')

    shapeA = Shape(vA, fA, name=nameA, rescale_unit_area=True)
    shapeB = Shape(vB, fB, name=nameB, rescale_unit_area=True)

    # load initial map
    S_init = np.genfromtxt(readPath + nameA + '_' + nameB + '_initMap.txt').astype(int)
    initP = scipy.sparse.lil_matrix((shapeB.nVert, shapeA.nVert))
    initP[S_init, np.arange(shapeA.nVert)] = 1

    # compute correspondences using the elastic eigenmodes (elasticBasis)
    Solv = CorrespondenceSolver(shapeA, shapeB, kmin=kmin, kmax=kmax, bending_weight=bending_weight, elasticBasis=True,
                                LB=False)
    # compute final correspondences by an iterative procedure
    P, C = Solv.computeCorrespondence(P=initP)

    # convert mapping matrix to indices vA->vB[corr_ours]
    corr_ours = P.toarray()
    corr_ours = np.nonzero(corr_ours.T)[1]
    np.savetxt(writePath + shapeA.name + '_' + shapeB.name + '_ElasticBasisresult.txt', corr_ours, fmt='%d')
   
    if precise:
        # compute vertex-to-point map
        P_prec = Solv.preciseMap(C)
        np.save(writePath + shapeA.name +'_' + shapeB.name +'ElasticPrecisemap.npy', P_prec, allow_pickle = True)

    # use the eigenfunctions of LB operator as a comparison (this method corresponds to ZoomOut)
    Solv_LB = CorrespondenceSolver(shapeA, shapeB, kmin=kmin, kmax=kmax, LB=True)
    P_LB, C_LB = Solv_LB.computeCorrespondence(P=initP)

    # convert mapping matrix to indices vA->vB[corr_ours]
    corr_LB = P_LB.toarray()
    corr_LB = np.nonzero(corr_LB.T)[1]
    np.savetxt(writePath + shapeA.name + '_' + shapeB.name + '_LBBasisresult.txt', corr_LB, fmt='%d')

    if precise:
        # compute vertex-to-point map
        P_prec_LB = Solv_LB.preciseMap(C_LB)
        np.save(writePath + shapeA.name +'_' + shapeB.name +'_LBPrecisemap.npy', P_prec_LB, allow_pickle = True)

    print("saved results in results folder")

    ########visualize results#######
    ps.init()
    source_mesh = ps.register_surface_mesh("source shape", shapeA.v, shapeA.f, smooth_shade=True)
    target_mesh = ps.register_surface_mesh("target shape", shapeB.v, shapeB.f, smooth_shade=True)

    # normal transfer
    target_mesh.add_color_quantity("normals", shapeB.normals, enabled=True)
    source_mesh.add_color_quantity("elastic Basis pullback normals", shapeB.normals[corr_ours], enabled=True)
    source_mesh.add_color_quantity("LB Basis pullback normals", shapeB.normals[corr_LB], enabled=False)

    target_mesh.set_position(np.array([0, 0, 1.5]))
    if precise:
        source_mesh.add_color_quantity("elastic Basis precise map pullback normals", P_prec.dot(shapeB.normals),
                                       enabled=True)
        source_mesh.add_color_quantity("LB basis precise map pullback normals", P_prec_LB.dot(shapeB.normals),
                                       enabled=False)

    ps.show()
