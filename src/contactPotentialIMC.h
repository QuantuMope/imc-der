#include <symengine/llvm_double.h>
#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"
#include "collisionDetector.h"

using namespace SymEngine;


class contactPotentialIMC
{
public:
    contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, collisionDetector &m_col_detector,
                        int m_ce_k, int m_friction, double m_mu, double m_vel_tol);

    void updateContactStiffness();
    void computeFc(bool first_iter);
    void computeFcJc(bool first_iter);
    double contact_stiffness;

private:
    elasticRod* rod;
    timeStepper* stepper;
    collisionDetector* col_detector;
    int ce_k;
    double h2;
    bool friction;
    double mu;
    double vel_tol;

    // Index helper
    vector<double> di{0, 1, 2, 4, 5, 6};

    Vector<double, 14> contact_input;
    Vector<double, 39> friction_input;
    Vector<double, 12> contact_gradient;
    Vector<double, 12> friction_forces;
    Matrix<double, 12, 12> contact_hessian;
//    Matrix<double, 12, 24> friction_jacobian_partials;
    Matrix<double, 12, 12> friction_partials_dfr_dx;
    Matrix<double, 12, 12> friction_partials_dfr_dfc;
    Matrix<double, 12, 12> friction_jacobian;

    void prepContactInput(int e1, int e2);
    void prepFrictionInput(int e1, int e2);

    LLVMDoubleVisitor contact_potential_gradient_func;
    LLVMDoubleVisitor contact_potential_hessian_func;
    LLVMDoubleVisitor friction_force_func;
    LLVMDoubleVisitor friction_partials_dfr_dx_func;
    LLVMDoubleVisitor friction_partials_dfr_dfc_func;

    void generatePotentialFunctions();

    // Helper functions for symbolic differentiation process
    void approx_fixbound(const RCP<const Basic> &input, RCP<const Basic> &result, const double &k);
    void approx_boxcar(const RCP<const Basic> &input, RCP<const Basic> &result, const double &k);
    void subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
};
