import numpy as np
import igl
import utils
import matplotlib.pyplot as plt
import os

from correspondence import Shape
from correspondence import CorrespondenceSolver

try:
    ps = __import__('polyscope')
    print('Visualization with polyscope.')
    visualization = True
except ImportError:
    print("Module polyscope not found. No visualization of result.")
    visualization = False

if __name__ == "__main__":
    readPath = 'data/'
    writePath = 'results/'
    if not os.path.exists(writePath):
        os.mkdir(writePath)

    # example1
    # nameA = 'headA'
    # nameB = 'headB'

    # example2
    nameA = 'homerA'
    nameB = 'homerB'

    bending_weight = 1e-5
    kmin = 5
    kmax = 100
    precise = False  # convert the final vertex map to a vertex-to-point map
    exists_gt = True  # groundtruth exists

    # load the groundtruth correspondences for measuring the error
    groundtruth = np.loadtxt(readPath + 'gt_' + nameA + '_' + nameB + '.txt').astype(int)

    # load 5 landmark correspondences
    landmarksA = np.loadtxt(readPath + nameA + '_landmarks.txt').astype(int)
    landmarksB = groundtruth[landmarksA]

    vA, textA, n, fA, m, b = igl.read_obj(readPath + nameA + '.obj')
    vB, textB, n, fB, m, b = igl.read_obj(readPath + nameB + '.obj')

    shapeA = Shape(vA, fA, name=nameA)
    shapeB = Shape(vB, fB, name=nameB)

    # compute correspondences using the elastic eigenmodes (elasticBasis)
    Solv_elastic = CorrespondenceSolver(shapeA, shapeB, kmin=kmin, kmax=kmax, bending_weight=bending_weight,
                                        elasticBasis=True)
    # create an initial functional map (kmin x kmin) by aligning the basis functions on landmarks
    C_init = Solv_elastic.CinitfromLandmarks(landmarksA, landmarksB)
    # compute final correspondences by an iterative procedure
    P, C = Solv_elastic.computeCorrespondence(C=C_init)

    # convert mapping matrix to indices vA->vB[corr_ours]
    corr_ours = P.toarray()
    corr_ours = np.nonzero(corr_ours.T)[1]
    np.savetxt(writePath + shapeA.name + '_' + shapeB.name + '_ElasticBasisresult.txt', corr_ours, fmt='%d')

    if precise:
        P_prec = Solv_elastic.preciseMap(C)
        np.save(writePath + shapeA.name +'_' + shapeB.name +'ElasticPrecisemap.npy', P_prec, allow_pickle = True)

    # use the eigenfunctions of LB operator as a comparison (this method corresponds to ZoomOut)
    Solv_LB = CorrespondenceSolver(shapeA, shapeB, kmin=kmin, kmax=kmax, LB=True)
    C_init = Solv_LB.CinitfromLandmarks(landmarksA, landmarksB)
    P_LB, C_LB = Solv_LB.computeCorrespondence(C=C_init)

    # #convert mapping matrix to indices vA->vB[corr_LB]
    corr_LB = P_LB.toarray()
    corr_LB = np.nonzero(corr_LB.T)[1]

    if precise:
        P_prec_LB = Solv_LB.preciseMap(C_LB)
        np.save(writePath + shapeA.name +'_' + shapeB.name +'_LBPrecisemap.npy', P_prec_LB, allow_pickle = True)

    np.savetxt(writePath + shapeA.name + '_' + shapeB.name + '_LBBasisresult.txt', corr_LB, fmt='%d')
    print('saved computed correspondence in result folder')
    if exists_gt:
        # compute geodesic error of final resulta

        distmatrix = utils.geodesic_distmat_dijkstra(shapeB.v, shapeB.f)
        distmatrix = distmatrix / np.sqrt(np.sum(shapeB.mass))

        # error of Ours
        dist = distmatrix[corr_ours, groundtruth]
        errOurs, percOurs = utils.compute_percentageError(dist, maxdist=0.1, step=0.01)

        # error of LB approach
        dist = distmatrix[corr_LB, groundtruth]
        errZO, percZO = utils.compute_percentageError(dist, maxdist=0.1, step=0.01)
        fig, ax = plt.subplots()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

        ax.set_ylabel('% correspondences', labelpad=10)
        ax.set_xlabel('geodesic error', labelpad=10)

        ax.set_xlim((0, 0.09))
        ax.set_ylim((0, 102))

        ax.plot(errOurs, percOurs * 100, c='red', label='elasticBasis', linewidth=3)
        ax.plot(errZO, percZO * 100, c='blue', label='LBBasis', linewidth=3)

        label_params = ax.get_legend_handles_labels()
        label_params[1].reverse()
        label_params[0].reverse()
        plt.grid()
        fig.legend()
        fig.savefig(writePath + 'errorPlot' + shapeA.name + '_' + shapeB.name)
        print("saved error plot in results folder")
    ########visualize results#######
    if visualization:
        ps.init()

        source_mesh = ps.register_surface_mesh("source shape", shapeA.v, shapeA.f, smooth_shade=True)
        target_mesh = ps.register_surface_mesh("target shape", shapeB.v, shapeB.f, smooth_shade=True)

        # normal transfer
        target_mesh.add_color_quantity("normals", shapeB.normals, enabled=True)
        source_mesh.add_color_quantity("elastic Basis pullback normals", shapeB.normals[corr_ours], enabled=True)
        source_mesh.add_color_quantity("LB Basis pullback normals", shapeB.normals[corr_LB], enabled=False)
        target_mesh.set_position(np.array([0, 0, 1]))

    if precise:
        source_mesh.add_color_quantity("elastic precise map pullback normals", P_prec.dot(shapeB.normals), enabled=True)
        source_mesh.add_color_quantity("LB precise map pullback normals", P_prec_LB.dot(shapeB.normals), enabled=False)

    ps.show()
