#ifndef COLLISION_H
#define COLLISION_H

#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"

extern zmq::socket_t socket;
extern double *contact_forces;
extern double *contact_hessian;
extern double *node_coordinates;

class collision
{
public:
    collision(elasticRod &m_rod, timeStepper &m_stepper);
    ~collision();

    void preparePythonSharedMemory(int iter);
    void computeFc();
    void computeJc();

private:
    elasticRod *rod;
    timeStepper *stepper;

};
#endif