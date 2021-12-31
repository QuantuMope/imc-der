#include <symengine/llvm_double.h>
#include "eigenIncludes.h"

using namespace SymEngine;


class symbolicEquations
{
public:
    symbolicEquations();

    void generateContactPotentialFunctions();
    void generateFrictionJacobianFunctions();
    void generateParallelContactPotentialFunctions();

    LLVMDoubleVisitor contact_potential_grad_func;
    LLVMDoubleVisitor contact_potential_hess_func;
    LLVMDoubleVisitor friction_partials_dfr_dx_func;
    LLVMDoubleVisitor friction_partials_dfr_dfc_func;
    LLVMDoubleVisitor parallel_ACBD_case_grad_func;
    LLVMDoubleVisitor parallel_ACBD_case_hess_func;
    LLVMDoubleVisitor parallel_ADBC_case_grad_func;
    LLVMDoubleVisitor parallel_ADBC_case_hess_func;
    LLVMDoubleVisitor parallel_CADB_case_grad_func;
    LLVMDoubleVisitor parallel_CADB_case_hess_func;
    LLVMDoubleVisitor parallel_DACB_case_grad_func;
    LLVMDoubleVisitor parallel_DACB_case_hess_func;

private:
    bool symbolic_cse;
    int opt_level;

    // Helper functions for symbolic differentiation process
    void approx_fixbound(const RCP<const Basic> &input, RCP<const Basic> &result, const double &k);
    void approx_boxcar(const RCP<const Basic> &input, RCP<const Basic> &result, const double &k);
    void subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C);

    RCP<const Basic> x1s_x;
    RCP<const Basic> x1s_y;
    RCP<const Basic> x1s_z;
    RCP<const Basic> x1e_x;
    RCP<const Basic> x1e_y;
    RCP<const Basic> x1e_z;
    RCP<const Basic> x2s_x;
    RCP<const Basic> x2s_y;
    RCP<const Basic> x2s_z;
    RCP<const Basic> x2e_x;
    RCP<const Basic> x2e_y;
    RCP<const Basic> x2e_z;
    RCP<const Basic> ce_k;
    RCP<const Basic> h2;

    RCP<const Basic> x1s_x0;
    RCP<const Basic> x1s_y0;
    RCP<const Basic> x1s_z0;
    RCP<const Basic> x1e_x0;
    RCP<const Basic> x1e_y0;
    RCP<const Basic> x1e_z0;
    RCP<const Basic> x2s_x0;
    RCP<const Basic> x2s_y0;
    RCP<const Basic> x2s_z0;
    RCP<const Basic> x2e_x0;
    RCP<const Basic> x2e_y0;
    RCP<const Basic> x2e_z0;
    RCP<const Basic> f1s_x;
    RCP<const Basic> f1s_y;
    RCP<const Basic> f1s_z;
    RCP<const Basic> f1e_x;
    RCP<const Basic> f1e_y;
    RCP<const Basic> f1e_z;
    RCP<const Basic> f2s_x;
    RCP<const Basic> f2s_y;
    RCP<const Basic> f2s_z;
    RCP<const Basic> f2e_x;
    RCP<const Basic> f2e_y;
    RCP<const Basic> f2e_z;
    RCP<const Basic> mu;
    RCP<const Basic> dt;
    RCP<const Basic> vel_tol;
};
