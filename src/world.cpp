#include "world.h"

world::world() {
    ;
}

world::world(setInput &m_inputData) {
    render = m_inputData.GetBoolOpt("render");                // boolean
    saveData = m_inputData.GetBoolOpt("saveData");            // boolean

    RodLength = m_inputData.GetScalarOpt("RodLength");         // meter
    helixradius = m_inputData.GetScalarOpt("helixradius");     // meter
    gVector = m_inputData.GetVecOpt("gVector");                      // m/s^2
    maxIter = m_inputData.GetIntOpt("maxIter");                // maximum number of iterations
    helixpitch = m_inputData.GetScalarOpt("helixpitch");       // meter
    rodRadius = m_inputData.GetScalarOpt("rodRadius");         // meter
    numVertices = m_inputData.GetIntOpt("numVertices");        // int_num
    youngM = m_inputData.GetScalarOpt("youngM");               // Pa
    Poisson = m_inputData.GetScalarOpt("Poisson");             // dimensionless
    deltaTime = m_inputData.GetScalarOpt("deltaTime");         // seconds
    tol = m_inputData.GetScalarOpt("tol");                     // small number like 10e-7
    stol = m_inputData.GetScalarOpt("stol");                   // small number, e.g. 0.1%
    density = m_inputData.GetScalarOpt("density");             // kg/m^3
    viscosity = m_inputData.GetScalarOpt("viscosity");         // viscosity in Pa-s
    pull_time = m_inputData.GetScalarOpt("pullTime");          // get time of pulling
    release_time = m_inputData.GetScalarOpt("releaseTime");    // get time of loosening
    wait_time = m_inputData.GetScalarOpt("waitTime");          // get time of waiting
    pull_speed = m_inputData.GetScalarOpt("pullSpeed");        // get speed of pulling
    ce_k = m_inputData.GetScalarOpt("ce_k");                   // contact energy stiffness
    col_limit = m_inputData.GetScalarOpt("col");               // distance limit for contact
    friction = m_inputData.GetIntOpt("friction");              // dynamic friction option
    mu = m_inputData.GetScalarOpt("mu");                       // friction coefficient
    vel_tol = m_inputData.GetScalarOpt("velTol");              // velocity tolerance for smooth friction
    line_search = m_inputData.GetIntOpt("lineSearch");         // flag for enabling line search
    knot_config = m_inputData.GetStringOpt("knotConfig");      // get initial knot configuration

    shearM = youngM / (2.0 * (1.0 + Poisson));                       // shear modulus

    alpha = 1.0;                                                 // newton step size
    total_iters = 0;                                             // total number of newton iterations

    totalTime = wait_time + pull_time + release_time;            // total sim time
}

world::~world() {
    ;
}

bool world::isRender() {
    return render;
}

void world::OpenFile(ofstream &outfile, string filename) {
    if (saveData == false) return;

    int systemRet = system("mkdir datafiles"); //make the directory
    if (systemRet == -1) {
        cout << "Error in creating directory\n";
    }

    // Open an input file named after the current time
    ostringstream name;
    name << "datafiles/" << filename << "_" << knot_config << "_" << to_string(pull_time) << ".txt";

    outfile.open(name.str().c_str());
    outfile.precision(10);
}

void world::outputNodeCoordinates(ofstream &outfile) {
     Vector3d curr_node;
     double curr_theta;
     for (int i = 0; i < rod->nv-1; i++) {
         curr_node = rod->getVertex(i);
         curr_theta = rod->getTheta(i);
         outfile << curr_node(0) << " " << curr_node(1) << " " <<
                    curr_node(2) << " " << curr_theta << endl;
     }
     curr_node = rod->getVertex(rod->nv-1);
     outfile << curr_node(0) << " " << curr_node(1) << " " <<
                curr_node(2) << " " << 0.0 << endl;
}

void world::CloseFile(ofstream &outfile) {
    if (saveData == false)
        return;

    outfile.close();
}


void world::CoutDataC(ofstream &outfile) {
    double f;
    double f1;

    f = temp.norm();
    f1 = temp1.norm();

    double end_to_end_length = (rod->getVertex(0) - rod->getVertex(numVertices - 1)).norm();

    // 2piR method
    double loop_circumference = 1.0 - end_to_end_length;
    double radius = loop_circumference / (2. * M_PI);

    // Output pull forces here
    // Do not need to add endl here as it will be added when time spent is added
    outfile << currentTime << " " << f << " " << f1 << " " << radius << " " << end_to_end_length
            << " " << iter << " " << total_iters << endl;
}


void world::setRodStepper() {
    // Set up geometry
    rodGeometry();

    // Create the rod
    rod = new elasticRod(vertices, vertices, density, rodRadius, deltaTime,
                         youngM, shearM, RodLength, theta);

    // Find out the tolerance, e.g. how small is enough?
    characteristicForce = M_PI * pow(rodRadius, 4) / 4.0 * youngM / pow(RodLength, 2);
    forceTol = tol * characteristicForce;

    // Set up boundary condition
    rodBoundaryCondition();

    // setup the rod so that all the relevant variables are populated
    rod->setup();
    // End of rod setup

    // set up the time stepper
    stepper = new timeStepper(*rod);
    totalForce = stepper->getForce();
    ls_nodes = new double[rod->ndof];
    dx = stepper->dx;

    // declare the forces
    m_stretchForce = new elasticStretchingForce(*rod, *stepper);
    m_bendingForce = new elasticBendingForce(*rod, *stepper);
    m_twistingForce = new elasticTwistingForce(*rod, *stepper);
    m_inertialForce = new inertialForce(*rod, *stepper);
    m_gravityForce = new externalGravityForce(*rod, *stepper, gVector);
    m_dampingForce = new dampingForce(*rod, *stepper, viscosity);
    m_collisionDetector = new collisionDetector(*rod, *stepper, col_limit);
    m_contactPotentialIMC = new contactPotentialIMC(*rod, *stepper, *m_collisionDetector, ce_k, friction, mu, vel_tol);

    // Allocate every thing to prepare for the first iteration
    rod->updateTimeStep();

    currentTime = 0.0;
}

// Setup geometry
void world::rodGeometry() {
    vertices = MatrixXd(numVertices, 3);

    ifstream myfile(("knot_configurations/" + knot_config).c_str());

    int row1 = numVertices;

    MatrixXd data = MatrixXd(row1, 4);
    double a;
    if (myfile.is_open()) {
        for (int i = 0; i < row1 * 4; i++) {
            myfile >> a;
            if (i % 4 == 0)
                data(i / 4, 0) = a;
            else if (i % 4 == 1)
                data(i / 4, 1) = a;
            else if (i % 4 == 2)
                data(i / 4, 2) = a;
            else if (i % 4 == 3)
                data(i / 4, 3) = a;
        }
    }
    theta = VectorXd::Zero(numVertices - 1);
    for (int i = 0; i < numVertices; i++) {
        vertices(i, 0) = data(i, 0);
        vertices(i, 1) = data(i, 1);
        vertices(i, 2) = data(i, 2);
    }
}


void world::rodBoundaryCondition() {
    // Knot tie boundary conditions
    rod->setVertexBoundaryCondition(rod->getVertex(0), 0);
    rod->setVertexBoundaryCondition(rod->getVertex(1), 1);
    rod->setThetaBoundaryCondition(0.0, 0);
    rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 1), numVertices - 1);
    rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 2), numVertices - 2);
    rod->setThetaBoundaryCondition(0.0, numVertices - 2);
}


void world::updateBoundary() {
    Vector3d u;
    u(0) = 0;
    u(1) = pull_speed;
    u(2) = 0;

    if (currentTime > wait_time && currentTime <= wait_time + pull_time) {   // Pulling
        rod->setVertexBoundaryCondition(rod->getVertex(0) - u * deltaTime, 0);
        rod->setVertexBoundaryCondition(rod->getVertex(1) - u * deltaTime, 1);
        rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 1) + u * deltaTime, numVertices - 1);
        rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 2) + u * deltaTime, numVertices - 2);
    } else if (currentTime > wait_time + pull_time &&
               currentTime <= wait_time + pull_time + release_time) {   // Loosening
        rod->setVertexBoundaryCondition(rod->getVertex(0) + u * deltaTime, 0);
        rod->setVertexBoundaryCondition(rod->getVertex(1) + u * deltaTime, 1);
        rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 1) - u * deltaTime, numVertices - 1);
        rod->setVertexBoundaryCondition(rod->getVertex(numVertices - 2) - u * deltaTime, numVertices - 2);
    }
}

void world::updateCons() {
    rod->updateMap();
    stepper->update();
    totalForce = stepper->getForce();
}


void world::updateTimeStep() {
    bool solved = false;

    updateBoundary();

    rod->updateGuess();  // our guess is just the previous position
    newtonMethod(solved);

    // calculate pull forces;
    calculateForce();

    rod->updateTimeStep();

    printSimData();

    currentTime += deltaTime;
}

void world::calculateForce() {
    stepper->setZero();

    m_inertialForce->computeFi();
    m_stretchForce->computeFs();
    m_bendingForce->computeFb();
    m_twistingForce->computeFt();
    m_gravityForce->computeFg();
    m_dampingForce->computeFd();

    temp[0] = stepper->force[0] + stepper->force[4];
    temp[1] = stepper->force[1] + stepper->force[5];
    temp[2] = stepper->force[2] + stepper->force[6];

    temp1[0] = stepper->force[rod->ndof - 3] + stepper->force[rod->ndof - 7];
    temp1[1] = stepper->force[rod->ndof - 2] + stepper->force[rod->ndof - 6];
    temp1[2] = stepper->force[rod->ndof - 1] + stepper->force[rod->ndof - 5];
}


bool world::pulling() {
    return currentTime > wait_time;
}


void world::newtonDamper() {
    if (iter < 10)
        alpha = 1.0;
    else
        alpha *= 0.90;
    if (alpha < 0.1)
        alpha = 0.1;
}


void world::printSimData() {
    printf("time: %.4f | iters: %i | con: %i | min_dist: %.6f | k: %.3e | fric: %i\n",
           currentTime, iter,
           m_collisionDetector->num_collisions,
           m_collisionDetector->min_dist,
           m_contactPotentialIMC->contact_stiffness,
           friction);
}


void world::newtonMethod(bool &solved) {
    double normf = forceTol * 10.0;
    double normf0 = 0;
    iter = 0;

    m_collisionDetector->detectCollisions();
    m_collisionDetector->detectParallelCases();

    while (solved == false) {
        rod->prepareForIteration();

        stepper->setZero();

        // Compute the forces and the jacobians
        m_inertialForce->computeFi();
        m_inertialForce->computeJi();

        m_stretchForce->computeFs();
        m_stretchForce->computeJs();

        m_bendingForce->computeFb();
        m_bendingForce->computeJb();

        m_twistingForce->computeFt();
        m_twistingForce->computeJt();

        m_gravityForce->computeFg();
        m_gravityForce->computeJg();

        m_dampingForce->computeFd();
        m_dampingForce->computeJd();

        if (iter == 0) {
            m_contactPotentialIMC->updateContactStiffness();
        }

        m_contactPotentialIMC->computeFcJc(currentTime == 0);

        // Compute norm of the force equations.
        normf = 0;
        for (int i = 0; i < rod->uncons; i++) {
            normf += totalForce[i] * totalForce[i];
        }
        normf = sqrt(normf);


        if (iter == 0) {
            normf0 = normf;
        }

        if (normf <= forceTol || (iter > 0 && normf <= normf0 * stol)) {
            solved = true;
            iter++;
            if (pulling())
                total_iters++;
        }

        if (solved == false) {
            stepper->integrator(); // Solve equations of motion
            if (line_search) {
                lineSearch();
            } else {
                newtonDamper();
            }
            rod->updateNewtonX(dx, alpha); // new q = old q + Delta q
            iter++;
            if (pulling())
                total_iters++;
        }

        // Exit if unable to converge
        if (iter > maxIter) {
            cout << "No convergence after " << maxIter << " iterations" << endl;
            exit(1);
        }
    }
}

int world::simulationRunning() {
    if (currentTime < totalTime)
        return 1;
    else {
        cout << "Completed simulation." << endl;
        return -1;
    }
}

int world::numPoints() {
    return rod->nv;
}

double world::getScaledCoordinate(int i) {
    return rod->x[i] / (0.325 * RodLength);
}

double world::getCurrentTime() {
    return currentTime;
}

void world::lineSearch() {
    // store current x
    rod->xold = rod->x;
    // Initialize an interval for optimal learning rate alpha
    double amax = 2;
    double amin = 1e-3;
    double al = 0;
    double au = 1;

    double a = 1;

    //compute the slope initially
    double q0 = 0.5 * pow(stepper->Force.norm(), 2);
    double dq0 = -(stepper->Force.transpose() * stepper->Jacobian * stepper->DX)(0);

    bool success = false;
    double m2 = 0.9;
    double m1 = 0.1;
    int iter_l = 0;
    while (!success) {
        rod->x = rod->xold;
        rod->updateNewtonX(dx, a);

        rod->prepareForIteration();

        stepper->setZero();

        // Compute the forces and the jacobians
        m_inertialForce->computeFi();
        m_stretchForce->computeFs();
        m_bendingForce->computeFb();
        m_twistingForce->computeFt();
        m_gravityForce->computeFg();
        m_dampingForce->computeFd();
        m_contactPotentialIMC->computeFc(currentTime == 0);

        double q = 0.5 * pow(stepper->Force.norm(), 2);

        double slope = (q - q0) / a;

        // cout << slope<<" "<<dq0<<" "<<a<<endl;

        if (slope >= m2 * dq0 && slope <= m1 * dq0) {
            success = true;
        } else {
            if (slope < m2 * dq0) {
                al = a;
            } else {
                au = a;
            }

            if (au < amax) {
                a = 0.5 * (al + au);
            } else {
                a = 10 * a;
            }
        }
        if (a > amax || a < amin) {
            break;
        }
        if (iter_l > 20) {
            break;
        }
        iter_l++;
    }
    rod->x = rod->xold;

    alpha = a;
}
