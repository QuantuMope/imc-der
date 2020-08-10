Compile and build:
------------------

Instructions for Ubuntu:
(1) To run this code you need Eigen, OpenGL, Pardiso and Lapack. Lapack is usually preinstalled on your computer. 
Eigen can be found at http://eigen.tuxfamily.org/index.php?title=Main_Page

(2) Open a terminal, "cd" to this folder and run the command "make" (without the quotes).

(3) To start the simulation, run the command "make run".


Physical parameters:
------------------

(1) You can edit the parameters of the simulation by editing "option.txt" file (recommended) and run using "make run"
    You can also specify an option using the following syntax:
    ./simDER option.txt -- option_name option_value
    Example: ./simDER option.txt -- RodLength 0.2

(2) Details on the options (we use SI units): 
    "RodLength" is the contour length of the helix.
    "helixradius" is the radius of the helix.
    "helixpitch" is the pitch of the helix.
    "rodRadius" is the cross-sectional radius of the flagellum.
    "youngM" is the young's modulus.
    "Poisson" is the Poisson ratio.
    "tol" and "stol" are small numbers used in solving the linear system. Fraction of a percent, e.g. 1.0e-3, is often a good choice.
    "maxIter" is the maximum number of iterations allowed before the solver quits.
    "density" is the mass per unit volume.
    "gVector" is the vector specifying acceleration due to gravity.
    "viscosity" is the viscosity of the fluid medium during pull time
    "deltaTime" is the time step size
    "numVertices" is the number of nodes on the rod.
    "render" (0 or 1) indicates whether OpenGL visualization should be displayed.
    "saveData" (0 or 1) indicates whether the location of the head should be saved in "datafiles/" folder (this folder will be created by the program).
    "recordNodes" (0 or 1) indicates whether or not nodes will be recorded
    "recordNodesStart" start time where nodes are recorded (ignored if record_nodes is 0)
    "recordNodesEnd" end time where nodes are recorded (ignored if record_nodes is 0)
    "waitTime" is the initial wait period duration
    "pullTime" is the duration to pull for (this starts after waitTime is done)
    "releaseTime" is the duration to loosen for (this starts after waitTime + pullTime is done)
    "friction" (0 or 1) indicates whether or not friction will be activated
    "port" the port number for shared memory
    "col" the collision limit
    "con" initial contact stiffness (multiplier for force and hessian)
    "ce_k" contact energy curve stiffness
    "mu_k" kinetic friction coefficient
    "knotConfig" knot configuration file name, this should be a txt file located in the knot_configurations directory
    "contactMode" (0 or 1 or 2) indicates the algorithm type, (0: no hessian w/ dgb, 1: hessian w/ pardiso, 2: hybrid)
    "limit" number of iterations before algorithm switches to hessian mode (ignored if contact_mode is not 2)
    "S" scaling factor. This value should equal: h2 / (rodRadius * 2) where h2 is the radius used to generate the gradient and hessian functions (usually 2.0)