/**
 * simDER
 * simDER stands for "[sim]plified [D]iscrete [E]lastic [R]ods"
 * Dec 2017
 * This code is based on previous iterations. 
 * */

//This line is for mac
//#include <GLUT/glut.h>

//This is for linux
#include <GL/glut.h>

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "eigenIncludes.h"

// Rod and stepper are included in the world
#include "world.h"
#include "setInput.h"

world myWorld;
int NPTS;
ofstream pull_data;
ofstream node_data;

// Setup ZMQ socket connection.
zmq::context_t context(1);
zmq::socket_t socket(context, ZMQ_REQ);

pid_t pid;

double* contact_forces;
double* contact_hessian;
double* node_coordinates;
double* velocities;
double* meta_data;

clock_t start;
clock_t finish;
double time_taken;

int record_nodes;
double record_nodes_start;
double record_nodes_end;

void kill_server(int sig_nm) {
    kill(pid, SIGTERM); // kill child before exiting
}

static void Key(unsigned char key, int x, int y)
{
  switch (key) // ESCAPE to quit
  {
    case 27:
        kill(pid, SIGTERM);  // kill child before exiting
        exit(0);
  }
}

/* Initialize OpenGL Graphics */
void initGL() 
{
    glClearColor(0.7f, 0.7f, 0.7f, 0.0f);  // Set background color to black and opaque
    glClearDepth(10.0f);                   // Set background depth to farthest
    glShadeModel(GL_SMOOTH);               // Enable smooth shading

    glLoadIdentity();
    gluLookAt(0.05, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    glPushMatrix();
}

void simulate() {
    start = std::clock();
    myWorld.updateTimeStep(); // update time step
    finish = std::clock();

    if (myWorld.pulling())
    {
        // Record contact data
        myWorld.CoutDataC(pull_data);

        // Record time taken to complete one time step
        time_taken = double(finish - start) / double(CLOCKS_PER_SEC);
        pull_data << time_taken << endl;

        // Record nodes for threejs
        if (record_nodes) {
            if (record_nodes_start < myWorld.getCurrentTime() && myWorld.getCurrentTime() < record_nodes_end) {
                myWorld.outputNodeCoordinates(node_data);
            }
        }
    }
}

void display(void)
{
    double currentTime  = 0;
    while ( myWorld.simulationRunning() > 0)
    {
        //  Clear screen and Z-buffer
        glClear(GL_COLOR_BUFFER_BIT);

        // draw axis
        double axisLen = 1;
        glLineWidth(0.5);

        glBegin(GL_LINES);

        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(axisLen, 0.0, 0.0);

        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, axisLen, 0.0);

        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, axisLen);

        glEnd();

        //draw a line
        glColor3f(0.1, 0.1, 0.1);
        glLineWidth(4.0);

        glBegin(GL_LINES);
        for (int i=0; i < NPTS-1; i++)
        {
            glVertex3f( myWorld.getScaledCoordinate(4*i), myWorld.getScaledCoordinate(4*i+1), myWorld.getScaledCoordinate(4*i+2));
            glVertex3f( myWorld.getScaledCoordinate(4*(i+1)), myWorld.getScaledCoordinate(4*(i+1)+1), myWorld.getScaledCoordinate(4*(i+1)+2));
        }
        glEnd();

        glFlush();

        simulate();
    }
    myWorld.CloseFile(pull_data);
    kill(pid, SIGTERM); // kill child before exiting
    exit(0);
}

int main(int argc,char *argv[])
{
    setInput inputData;
    inputData = setInput();
    inputData.LoadOptions(argv[1]);
    inputData.LoadOptions(argc,argv);

    // Setup shared memory
    int num_nodes = inputData.GetIntOpt("numVertices");
    int nv = num_nodes * 3;
    int hess_size = nv * nv;
    int meta_data_size = 6;
    string port_no = to_string(inputData.GetIntOpt("port"));

    int nc_fd = shm_open(("node_coordinates" + port_no).c_str(), O_CREAT | O_RDWR, 0666);
    int ve_fd = shm_open(("velocities" + port_no).c_str(), O_CREAT | O_RDWR, 0666);
    int me_fd = shm_open(("meta_data" + port_no).c_str(), O_CREAT | O_RDWR, 0666);
    int cf_fd = shm_open(("contact_forces" + port_no).c_str(), O_CREAT | O_RDWR, 0666);
    int ch_fd = shm_open(("contact_hessian" + port_no).c_str(), O_CREAT | O_RDWR, 0666);

    if (nc_fd == -1 || ve_fd == -1 || me_fd == -1 || cf_fd == -1 || ch_fd == -1)
    {
        cout << "Failed to open shared memory for writing" << endl;
        exit(1);
    }
    if (ftruncate(nc_fd, sizeof(double)*nv) == -1 ||
        ftruncate(ve_fd, sizeof(double)*nv) == -1 ||
        ftruncate(me_fd, sizeof(double)*meta_data_size) == -1 ||
        ftruncate(cf_fd, sizeof(double)*nv) == -1 ||
        ftruncate(ch_fd, sizeof(double)*hess_size) == -1)
    {
        cout << "Failed to truncate shared memory" << endl;
        exit(1);
    }
    node_coordinates = (double*)mmap(NULL, sizeof(double)*nv, PROT_WRITE, MAP_SHARED, nc_fd, 0);
    velocities = (double*)mmap(NULL, sizeof(double)*nv, PROT_WRITE, MAP_SHARED, ve_fd, 0);
    meta_data = (double*)mmap(NULL, sizeof(double)*meta_data_size, PROT_WRITE, MAP_SHARED, me_fd, 0);
    contact_forces = (double*)mmap(NULL, sizeof(double)*nv, PROT_READ, MAP_SHARED, cf_fd, 0);
    contact_hessian = (double*)mmap(NULL, sizeof(double)*hess_size, PROT_READ, MAP_SHARED, ch_fd, 0);

    if (node_coordinates == MAP_FAILED ||
        velocities == MAP_FAILED ||
        meta_data == MAP_FAILED ||
        contact_forces == MAP_FAILED ||
        contact_hessian == MAP_FAILED)
    {
        cout << "Failed to map shared memory for writing" << endl;
        exit(1);
    }

    //read input parameters from txt file and cmd
    myWorld = world(inputData);
    myWorld.setRodStepper();

    myWorld.OpenFile(pull_data, "pull_data");
    record_nodes = inputData.GetIntOpt("recordNodes");
    record_nodes_start = inputData.GetScalarOpt("recordNodesStart");
    record_nodes_end = inputData.GetScalarOpt("recordNodesEnd");
    if (record_nodes) {
        myWorld.OpenFile(node_data, "node_data");
    }

    const char* port = to_string(inputData.GetIntOpt("port")).c_str();
    const char* col = to_string(inputData.GetScalarOpt("col")).c_str();
    const char* con = to_string(inputData.GetScalarOpt("con")).c_str();
    const char* ce_k = to_string(inputData.GetScalarOpt("ce_k")).c_str();
    const char* mu_k = to_string(inputData.GetScalarOpt("mu_k")).c_str();
    const char* S = to_string(inputData.GetScalarOpt("S")).c_str();
    const char* radius = to_string(inputData.GetScalarOpt("rodRadius")).c_str();
    const char* nv_py = to_string(inputData.GetIntOpt("numVertices")).c_str();

    cout << "Input parameters to python" << endl;
    cout << "port " << port << " col " << col << " con " << con << " ce_k " << ce_k << endl;
    cout << "mu_k " << mu_k << " rod radius " << radius << " number of vertices " << nv_py << endl;

    // Start up Python Server.
    if ((pid = fork()) < 0) {
        cout << "Forking python server failed." << endl;
        exit(1);
    }
    else if (pid == 0) {
        int server_result = execlp("python3", "python3", "../imc.py", port, col, con, ce_k,
                                   mu_k, radius, nv_py, S, (char*)NULL);
        if (server_result < 0) {
            cout << "Failed to initialize python server." << endl;
            exit(1);
        }
    }
    // If ctrl+C or segfault, kill python server.
    signal(SIGINT, kill_server);
    signal(SIGSEGV, kill_server);

    // Connect to Python ZMQ
    string server_addr = "tcp://localhost:";
    cout << "Connecting to Python server..." << endl;
    socket.connect(server_addr + port);

    bool render = myWorld.isRender();
    if (render) // if OpenGL visualization is on
    {
        NPTS = myWorld.numPoints();

        glutInit(&argc,argv);
        glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
        glutInitWindowSize (1000, 1000);
        glutInitWindowPosition (100, 100);
        glutCreateWindow ("simDER");
        initGL();
        glutKeyboardFunc(Key);
        glutDisplayFunc(display);
        glutMainLoop();
    }
    else
    {
        while ( myWorld.simulationRunning() > 0)
        {
            simulate();
        }
    }

    // Close (if necessary) the data file
    myWorld.CloseFile(pull_data);
    kill(pid, SIGTERM); // kill child before exiting
    exit(0);
}

