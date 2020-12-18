#include "dampingForce.h"
#include <iostream>

dampingForce::dampingForce(elasticRod &m_rod, timeStepper &m_stepper, double m_viscosity)
{
    rod = &m_rod;
    stepper = &m_stepper;
    viscosity = m_viscosity;

    Id3 << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;

}

dampingForce::~dampingForce()
{
    ;
}

void dampingForce::computeFd()
{
    for (int i = 0; i < rod->ne; i++)
    {
        force = -viscosity * (rod->getVertex(i) - rod->getPreVertex(i))  / rod->dt * rod->voronoiLen(i);
        for (int k = 0; k < 3; k++)
        {
            ind = 4 * i + k;
            stepper->addForce(ind, - force[k]); // subtracting external force
        }
    }
}

void dampingForce::computeJd()
{
    for (int i = 0; i < rod->ne; i++)
    {
        jac = -viscosity * rod->voronoiLen(i) / rod->dt * Id3;

        for (int kx = 0; kx < 3; kx++)
        {
            indx = 4 * i + kx;
            for (int ky = 0; ky < 3; ky++)
            {
                indy = 4 * i + ky;
                stepper->addJacobian(indx, indy, - jac(kx,ky)); // subtracting external force
            }
        }
    }
}
//
//void dampingForce::computeFd()
//{
//    for (int i = 0; i < rod->ne; i++)
//    {
//        force = -viscosity * rod->getVelocity(i) * rod->voronoiLen(i);
//        for (int k = 0; k < 3; k++)
//        {
//            ind = 4 * i + k;
//            stepper->addForce(ind, - force[k]); // subtracting external force
//        }
//    }
//}
//
//void dampingForce::computeJd()
//{
//    for (int i = 0; i < rod->ne; i++)
//    {
//        jac = -viscosity * rod->voronoiLen(i) / rod->dt * Id3;
//
//        for (int kx = 0; kx < 3; kx++)
//        {
//            indx = 4 * i + kx;
//            for (int ky = 0; ky < 3; ky++)
//            {
//                indy = 4 * i + ky;
//                stepper->addJacobian(indx, indy, - jac(kx,ky)); // subtracting external force
//            }
//        }
//    }
//}
