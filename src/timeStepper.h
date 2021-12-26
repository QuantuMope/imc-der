#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

#include "elasticRod.h"

/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);


class timeStepper
{
public:
    timeStepper(elasticRod &m_rod);
    ~timeStepper();
    double* getForce();
    double* getJacobian();
    void setZero();
    void addForce(int ind, double p);
    void addJacobian(int ind1, int ind2, double p);
    void integrator();

    void pardisoSolver();

    VectorXd force;

    void update();
    VectorXd Force;
    MatrixXd Jacobian;
    VectorXd DX;
    double *dx;

private:
    elasticRod *rod;
    int kl, ku, freeDOF;

    double *totalForce;
    double *jacobian;

    // utility variables
    int mappedInd, mappedInd1, mappedInd2;
    int row, col, offset;
    int NUMROWS;
    int jacobianLen;
    int nrhs;
    int *ipiv;
    int info;
    int ldb;
};

#endif
