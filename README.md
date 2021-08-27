## Implicit Contact Method (IMC)
Contact method for 3D elastic rod simulations. Uses Discrete Elastic Rod (DER) framework and incorporates contact and friction. Formulates a contact potential as a twice differentiable analytical expression 
and uses the subsequent gradient (Forces) and Hessian (Jacobian) to simulate contact. The published method can be found [here](misc/imc_paper.pdf) along with a [graphical abstract](https://www.youtube.com/watch?v=yq4-m0G0D4g&feature=youtu.be).

<p align="center">
<img src="images/knot_tying.png" alt>
<br>
<em> Figure 1. Simulation examples for tying overhand knots with various unknotting numbers. </em>
</p>

***
#### TODOs
- Optimize hessian chain rule procedure (more details in src/imc.py)
- Convert IMC code to C++

***
#### How to use
First, the necessary functions must be generated and stored. Use the following command line argument to
create the functions for a certain contact energy stiffness (ce_k) and scaled contact distance (h2, two times the radius).
```bash
cd src                                   # go to source code directory
python3 initialize_functions.py ce_k h2  # a good value is ce_k = 50.0 and h2 = 2.0
```
This should only take a few seconds since changing the symbolic differentiation from sympy to symengine. After the functions are generated once, the simulator
can be run with the following commands.
```bash
cd DER                 # go to the DER directory
make -j4               # compile the program
make run               # run the program
```
The option.txt contains all simulation settings. Please refer to the README.txt in the DER directory for a description
of all inputs.
Note that the simulation cannot run with settings for functions that have not yet been generated. Multiple ce_k, h2
functions can be created as they are not overwritten.
Lastly, in the Makefile, the Eigen library location should be specified.

All code tested and developed on **Ubuntu 18.04.4 LTS** using **Python 3.6.9** and **C++11**.

***
#### Dependencies
Run the following to download all dependencies. If Python version is not at least 3.6, do not install numba library.
Numba is used as a JIT compiler for several functions in the implicit contact method computation. This
allows for the speed of Python code to become more comparable to that of C++.
If at least Python 3.6 is not used, remove the @njit decorator in imc.py and imc_utils.py.
```bash
pip3 install numpy sympy symengine matplotlib dill posix_ipc numba
```
C++ dependencies are listed in the README.txt in the DER directory.
Furthermore, this program uses the Pardiso solver which can be found [here](https://www.pardiso-project.org/).
***
#### Dependencies for proper cross language data transfer
Primary data transfer technique used is shared memory using Posix IPC API. In addition this,
the following software tools are used as a synchronization and communication tool.
1. [ZeroMQ for Python](https://zeromq.org/languages/python/)
2. [ZeroMQ for C/C++](https://github.com/zeromq/libzmq)
3. [cppzmq: ZMQ C++ Header-only binding](https://github.com/zeromq/cppzmq)

***
#### How to install
a. ZMQ for Python can simply be installed using the pip command 
```bash
pip install pyzmq
```
b. Download libzmq from github and unzip the library. cd into the directory
```bash
mkdir build
cd build
cmake ..
sudo make -j4 install
```
c. Do the exact same steps as b. but for cppzmq.
