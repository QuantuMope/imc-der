#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"
#include "collisionDetector.h"
#include "symbolicEquations.h"


class contactPotentialIMC
{
public:
    contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, collisionDetector &m_col_detector,
                        double m_delta, double m_mu, double m_nu);

    void updateContactStiffness();
    void computeFc(bool waitTime);
    void computeFcJc(bool waitTime);
    double contact_stiffness;

private:
    elasticRod* rod;
    timeStepper* stepper;
    collisionDetector* col_detector;
    symbolicEquations* sym_eqs;
    double K1;
    double K2;
    double h2;
    double delta;
    bool friction;
    double mu;
    double nu;
    double scale;
    int fric_jaco_type;
    double velo_limit;

    // Index helper
    vector<double> di{0, 1, 2, 4, 5, 6};

    Vector<double, 8> p2p_input;
    Vector<double, 11> e2p_input;
    Vector<double, 14> e2e_input;

    Vector<double, 6> p2p_gradient;
    Vector<double, 9> e2p_gradient;
    Vector<double, 12> e2e_gradient;

    Matrix<double, 6, 6> p2p_hessian;
    Matrix<double, 9, 9> e2p_hessian;
    Matrix<double, 12, 12> e2e_hessian;

    Matrix<double, 3, 3> eye_mat;

    Vector<double, 39> friction_input;
    Vector<double, 17> friction_input2;
    Vector<double, 12> contact_gradient;
    Vector<double, 12> friction_forces;
    Matrix<double, 12, 12> contact_hessian;
    Matrix<double, 12, 12> friction_partials_dfr_dx;
    Matrix<double, 12, 12> friction_partials_dfr_dfc;
    Matrix<double, 12, 12> friction_jacobian;

    Matrix<double, 3, 12> dtv_rel_dfc;
    Matrix<double, 12, 3> dtv_rel_dfc_T;
    Matrix<double, 3, 12> dtv_rel_dx;
    Matrix<double, 12, 3> dtv_rel_dx_T;
    Vector<double, 1> dgamma_dtv_rel_n;
    Vector<double, 2> dgamma_input;

    Matrix<double, 12, 1> dfr_dgamma;
    Matrix<double, 12, 3> dfr_dtv_rel_u;
    Matrix<double, 3, 12> dfr_dtv_rel_u_T;
    Matrix<double, 12, 12> dfr_dfc;

    Matrix<double, 3, 12> dtv_rel_dx_cr;
    Matrix<double, 3, 12> dtv_rel_u_dx_cr;
    Matrix<double, 1, 12> dgamma_dx_cr;

    Matrix<double, 3, 1> tv_rel;
    Matrix<double, 3, 1> tv_rel_u;
    double tv_rel_n;

    void compute_dgamma_dx();
    void compute_dtv_rel_u_dx();

    void prepContactInput(int edge1, int edge2, int edge3, int edge4, int constraintType);
    void prepFrictionInput(int edge1, int edge2, int edge3, int edge4);
    void computeFriction(const int edge1,const int edge2,const int edge3,const int edge4);
};
