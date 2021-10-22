## [Implicit Contact Model (IMC)](https://asmedigitalcollection.asme.org/appliedmechanics/article/88/5/051010/1099667/Implicit-Contact-Model-for-Discrete-Elastic-Rods)

Contact model for 3D elastic rod simulations. Uses [Discrete Elastic Rod (DER)](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf) framework and incorporates contact and friction. Formulates a contact potential as a twice differentiable analytical expression through smooth approximations 
and uses the subsequent energy gradient (forces) and Hessian (force Jacobian) to simulate contact and friction. The published method can be found [here](misc/imc_paper.pdf) along with a [graphical abstract](https://www.youtube.com/watch?v=yq4-m0G0D4g&feature=youtu.be). Simulation examples using IMC to resolve contact and friction can be seen below in Figure 1.

**IMPORTANT NOTE!!!**: Recently an error was found in the Hessian computation for IMC! Performance of the fully implicit version has improved drastically and time steps much larger than the ones [reported](misc/imc_paper.pdf) are now possible. It is now much more preferable to use the fully implicit formulation over the hybrid algorithm.

<p align="center">
<img src="images/knot_tying.png" alt>
<br>
<em> Figure 1. Simulation examples for tying overhand knots with various unknotting numbers. </em>
</p>

***

## How to Use
Note that IMC is currently implemented purely in Python for ease of prototyping while the DER framework is implemented in C++. The performance gap between Python and C++ is mitigated by using almost purely numpy operations compiled through [numba](https://numba.pydata.org/) as well as symbolic differentation through [symengine](https://github.com/symengine/symengine) and function generation using LLVM.

All code tested and developed on **Ubuntu 18.04.4 LTS** using **Python 3.6.9** and **C++11**.
***

### Dependencies
#### Python Dependencies
Install the following Python dependencies. If Python version is not at least 3.6, do not install the numba library and remove the @njit decorator in ```imc_utils.py```. It is highly recommended to use use Python >= 3.6 to take advantage of numba.
```bash
python3 -m pip install numpy symengine numba dill posix_ipc pyzmq 
```
#### C++ Dependencies
Install the following C++ dependencies:
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Pardiso Solver](https://www.pardiso-project.org/)
- OpenGL
- Lapack (*usually preinstalled on your computer*)
- [libzmq](https://github.com/zeromq/libzmq)
- [cppzmq](https://github.com/zeromq/cppzmq)

***
### Compiling
First, the necessary functions must be generated and stored. Use the following command line argument to
create the functions for a certain contact energy stiffness ```ce_k``` and scaled contact distance ```h2``` (*two times the radius*).
```bash
cd src                                     # go to source code directory
python3 generate_functions.py $ce_k $h2    # a good value is ce_k = 50.0 and h2 = 2.0
```
This should only take a few seconds since changing the symbolic differentiation from sympy to symengine. 

Next, you will have to provide a Pardiso license file in the DER directory. This license can be obtained [here](https://www.pardiso-project.org/#download). Make sure all dependencies and locations are properly listed in ```Makefile``` and then compile DER.
```bash
cd DER                 # go to the DER directory
make                   # compile the program
```

***

### Parameters

All simulation parameters are set through a parameter file ```option.txt```. A template file ```template_option.txt``` is provided that can be used to construct ```option.txt```.

```bash
cp template_option.txt option.txt   # create option.txt
```
Specifiable parameters are as follows (we use SI units):
- ```RodLength``` - Contour length of the rod.
- ```numVertices``` - Number of nodes on the rod.
- ```rodRadius``` - Cross-sectional radius of the rod.
- ```helixradius``` - Radius of the helix.
- ```helixpitch``` - Pitch of the helix.
- ```density``` - Mass per unit volume.
- ```youngM``` - Young's modulus.
- ```Poisson``` - Poisson ratio.
- ```tol``` and ```stol``` - Small numbers used in solving the linear system. Fraction of a percent, e.g. 1.0e-3, is often a good choice.
- ```maxIter``` - Maximum number of iterations allowed before the solver quits. 
- ```gVector``` - 3x1 vector specifying acceleration due to gravity.
- ```viscosity``` - Viscosity of the fluid medium.
- ```render (0 or 1) ```- Flag indicating whether OpenGL visualization should be rendered.
- ```saveData (0 or 1)``` - Flag indicating whether pull forces and rod end positions should be reocrded.
- ```recordNodes (0 or 1)``` - Flag indicating whether nodal positions will be recorded.
- ```recordNodesStart``` - Start time for node recording (*ignored if ```recordNodes``` is 0*).
- ```recordNodesEnd``` - End time for node recording (*ignored if ```recordNodes``` is 0*).
- ```waitTime``` - Initial wait period duration.
- ```pullTime``` - Duration to pull for (*starts after ```waitTime``` is done*).
- ```releaseTime``` - Duration to loosen for (*starts after ```waitTime``` + ```pullTime``` is done*).
- ```pullSpeed``` - Speed at which to pull and/or loosen each end.
- ```deltaTime``` - Time step size.
- ```friction (0 or 1)``` - Flag indicating whether friction will be simulated.
- ```mu_k``` - Kinetic friction coefficient.
- ```col``` - Scaled collision limit.
- ```con``` - Initial contact stiffness.
- ```ce_k``` - Contact energy curve stiffness (*functions for this value must be pre-generated*).
- ```S``` - Scaling factor.
- ```knotConfig``` - File name for the initial knot configuration. Should be a txt file located in ```DER/knot_configurations``` directory.
- ```contactMode (0 or 1 or 2)``` - Flag indicating algorithm type, (0: explicit with dgb, 1: implicit with pardiso, 2: hybrid)
- ```limit``` - Number of iterations before algorithm switches to implicit (*ignored if ```contactMode``` is not 2*).

Once parameters are set. The simulation can be ran from a terminal by running ```make run```.

***

### Citation
If our work has helped your research, please cite the following paper.
```
article{10.1115/1.4050238,
    author = {Choi, Andrew and Tong, Dezhong and Jawed, Mohammad K. and Joo, Jungseock},
    title = "{Implicit Contact Model for Discrete Elastic Rods in Knot Tying}",
    journal = {Journal of Applied Mechanics},
    volume = {88},
    number = {5},
    year = {2021},
    month = {03},
}
```



