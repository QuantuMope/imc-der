#include "collision.h"
#include <bits/stdc++.h>

collision::collision(elasticRod &m_rod, timeStepper &m_stepper)
{
    rod = &m_rod;
    stepper = &m_stepper;
}

collision::~collision()
{
    ;
}

void collision::preparePythonSharedMemory(int iter)
{
    // write data to shared memory location
    // This is inefficient but is okay for now
    Vector3d node;
    if (iter == 0) {
        // Previous time step node coordinates only needs to be updated once.
        for (int i = 0; i < rod->nv; i++) {
            node = rod->getPreVertex(i);
            prev_node_coordinates[i*3] = node(0);
            prev_node_coordinates[(i*3)+1] = node(1);
            prev_node_coordinates[(i*3)+2] = node(2);
        }
    }
    for (int i = 0; i < rod->nv; i++) {
        node = rod->getVertex(i);
        node_coordinates[i*3] = node(0);
        node_coordinates[(i*3)+1] = node(1);
        node_coordinates[(i*3)+2] = node(2);
    }

    // Lock and unlock
    zmq::message_t request(0);
    socket.send(request);

    // Block until receiving force and hessian
    zmq::message_t reply;
    socket.recv(&reply);
}


void collision::computeFc()
{
    // give values to solver
    int force_index = 0;
    for (int i=0; i < rod->ndof; i++)
    {
        if ((i+1) % 4 == 0) continue;
        if (contact_forces[force_index] != 0)
            stepper->addForce(i, contact_forces[force_index]);
        force_index++;
    }
}

void collision::computeJc()
{
    // give values of jacobian_c to solver
    int jacob_index = 0;
    for (int i = 0; i < rod->ndof; i++)
    {
        if ((i+1) % 4 == 0) continue;
        for (int j = 0; j < rod->ndof; j++)
        {
            if ((j+1) % 4 == 0) continue;
            if (contact_hessian[jacob_index] != 0)
                stepper->addJacobian(j, i, contact_hessian[jacob_index]);
            jacob_index++;
        }
    }
}
