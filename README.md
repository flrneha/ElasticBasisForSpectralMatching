![We compute correspondences between shapes by computing a functional
map using (projected) eigenfunctions of an elastic shell energy’s Hessian as a functional basis. These basis functions are sensitive to extrinsic features such as extremities and creases (left). Hence, using them in a functional map pipeline enables us to accurately align crease lines such as mouth, ears, and toes (right). Here, we visualize the resulting correspondence by a pullback of normals from the ferret (top) to the weasel (bottom). The (transferred) normal directions are mapped to colors as shown on the little sphere.](images/teaser.png)


# An Elastic Basis for Spectral Shape Correspondence
This code accompanies the SIGGRAPH 2023 conference paper [An Elastic Basis for Spectral Shape Correspondence](https://doi.org/10.1145/3588432.3591518), by [Florine Hartwig](https://ins.uni-bonn.de/staff/hartwig), [Josua Sassen](https://josuasassen.com/), [Omri Azencot](https://omriazencot.com/), [Martin Rumpf](https://ins.uni-bonn.de/staff/rumpf
), [Mirela Ben-Chen](https://mirela.net.technion.ac.il/)

In this paper we develop a spectral non-isometric correspondence method that aligns extrinsic features using a functional map approach. We propose a novel crease-aware spectral basis derived from the Hessian of an elastic thin shell energy and describe the necessary adaptations for using non-orthogonal basis functions in the functional map framework.

## Requirements

This code uses python bindings for an implementation of the Discrete Shell Energy available here
- [Thin shell energy](https://gitlab.com/numod/shell-energy)

and the following packages
- `numpy`
- `scipy.sparse`
- `scipy.spatial`
- [libigl](https://libigl.github.io/libigl-python-bindings/)
- [polyscope](https://polyscope.run/py/) (for visualization)
- `tqdm`
- `matplotlib`

## Demo Scripts
We provide two example scripts which show the basis functionality of the code. For comparison, our code also provides the option to use the eigenfunctions of the Laplace Beltrami operator as a basis for the functional map approach. In this case our method reduces to [ZoomOut](https://github.com/llorz/SGA19_zoomOut).
- `Cartoon.py` script to compute correspondences between modified versions of a [Homer](http://visionair.ge.imati.cnr.it/ontologies/shapes/view.jsp?id=30-Homer) and a [Max-Planck bust](http://visionair.ge.imati.cnr.it/ontologies/shapes/view.jsp?id=77-Max-Planck_bust) model. The necessary data is stored in `data/` . For creating the deformed mesh versions, we used an implementation of *Computational caricaturization of surfaces* by Sela, Matan, Yonathan Aflalo, and Ron Kimmel. Computer Vision and Image Understanding 141 (2015): 1-17. 
- `CatLion.py` script to compute correspondences between a cat and lion shape available at http://people.csail.mit.edu/sumner/research/deftransfer/data.html. We provide a [shell script](downloadCatLionData.sh) which will automatically download and store the necessary data (cat-reference.obj and lion-reference.obj). 

For more details and theoretic background have a look at our paper. If you should have any questions, feel free to reach out to [Florine Hartwig](https://ins.uni-bonn.de/staff/hartwig)!
## Attribution
Please cite our paper when using this code. You can use the following bibtex

```
@unpublished{HaSaAz23,
author = {Hartwig, Florine and Sassen, Josua and Azencot, Omri and Rumpf, Martin and Ben-Chen, Mirela},
title = {{An Elastic Basis for Spectral Shape Correspondence}},
year = {2023},
note = {accepted as SIGGRAPH Conference Paper}
}
```
