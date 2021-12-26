#include "contactPotentialIMC.h"
#include <fstream>

using namespace SymEngine;
using namespace std;


contactPotentialIMC::contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, collisionDetector &m_col_detector,
                                         int m_ce_k, int m_friction, double m_mu, double m_vel_tol) {
    rod = &m_rod;
    stepper = &m_stepper;
    col_detector = &m_col_detector;

    ce_k = m_ce_k;
    h2 = rod->rodRadius * 2;
    friction = m_friction;
    mu = m_mu;
    vel_tol = m_vel_tol;

    // Setup constant inputs
    contact_input[12] = ce_k;
    contact_input[13] = h2;

    friction_input[36] = mu;
    friction_input[37] = rod->dt;
    friction_input[38] = vel_tol;

    contact_gradient.setZero();
    friction_forces.setZero();
    contact_hessian.setZero();
    friction_partials_dfr_dx.setZero();
    friction_partials_dfr_dfc.setZero();
    friction_jacobian.setZero();

    generatePotentialFunctions();
}


void contactPotentialIMC::approx_fixbound(const RCP<const Basic> &input, RCP<const Basic> &result, const double &m_k) {
    const RCP<const RealDouble> k = real_double(m_k);
    auto first_term = log(add(one, exp(mul(k, input))));
    auto second_term = log(add(one, exp(mul(k, sub(input, one)))));
    result = mul(div(one, k), sub(first_term, second_term));
}


void contactPotentialIMC::approx_boxcar(const RCP<const Basic> &input, RCP<const Basic> &result, const double &m_k) {
    const RCP<const RealDouble> k = real_double(m_k);
    auto first_term = div(one, add(one, exp(mul(mul(integer(-1), k), input))));
    auto second_term = div(one, add(one, exp(mul(mul(integer(-1), k), sub(input, one)))));
    result = sub(first_term, second_term);
}


// For some reason SymEngine doesn't have this implemented X_X
void contactPotentialIMC::subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
    assert((A.nrows() == B.nrows()) && (A.ncols() == B.ncols()));
    for (unsigned i=0; i < A.nrows(); i++) {
        for (unsigned j=0; j < A.ncols(); j++) {
            C.set(i, j, sub(A.get(i, j), B.get(i, j)));
        }
    }
}


void contactPotentialIMC::updateContactStiffness() {
    double curr_max_force = -1;
    double curr_force;
    double fx, fy, fz;
    set<int> nodes_to_check;

    // Compute the maximum force that a node experiences.
    for (int i = 0; i < col_detector->num_collisions; i++) {
        nodes_to_check.insert(col_detector->candidate_ids(i, 0));
        nodes_to_check.insert(col_detector->candidate_ids(i, 0)+1);
        nodes_to_check.insert(col_detector->candidate_ids(i, 1));
        nodes_to_check.insert(col_detector->candidate_ids(i, 1)+1);
    }

    for (auto i : nodes_to_check) {
        if (rod->getIfConstrained(i) == 0) {
            fx = stepper->getForce()[rod->fullToUnconsMap[4*i]];
            fy = stepper->getForce()[rod->fullToUnconsMap[4*i+1]];
            fz = stepper->getForce()[rod->fullToUnconsMap[4*i+2]];
        }
        else {
            continue;
        }
        curr_force = pow(pow(fx, 2) + pow(fy, 2) + pow(fz, 2), 0.5);
        if (curr_force > curr_max_force) {
            curr_max_force = curr_force;
        }
    }
    contact_stiffness = 2 * curr_max_force;
}


void contactPotentialIMC::prepContactInput(int edge1, int edge2) {
    Vector3d x1s = rod->getVertex(edge1);
    Vector3d x1e = rod->getVertex(edge1+1);
    Vector3d x2s = rod->getVertex(edge2);
    Vector3d x2e = rod->getVertex(edge2+1);

    contact_input[0] = x1s(0);
    contact_input[1] = x1s(1);
    contact_input[2] = x1s(2);
    contact_input[3] = x1e(0);
    contact_input[4] = x1e(1);
    contact_input[5] = x1e(2);
    contact_input[6] = x2s(0);
    contact_input[7] = x2s(1);
    contact_input[8] = x2s(2);
    contact_input[9] = x2e(0);
    contact_input[10] = x2e(1);
    contact_input[11] = x2e(2);
}


void contactPotentialIMC::prepFrictionInput(int edge1, int edge2) {
    Vector3d x1s = rod->getVertex(edge1);
    Vector3d x1e = rod->getVertex(edge1+1);
    Vector3d x2s = rod->getVertex(edge2);
    Vector3d x2e = rod->getVertex(edge2+1);
    Vector3d x1s0 = rod->getPreVertex(edge1);
    Vector3d x1e0 = rod->getPreVertex(edge1+1);
    Vector3d x2s0 = rod->getPreVertex(edge2);
    Vector3d x2e0 = rod->getPreVertex(edge2+1);

    friction_input[0] = x1s(0);
    friction_input[1] = x1s(1);
    friction_input[2] = x1s(2);
    friction_input[3] = x1e(0);
    friction_input[4] = x1e(1);
    friction_input[5] = x1e(2);
    friction_input[6] = x2s(0);
    friction_input[7] = x2s(1);
    friction_input[8] = x2s(2);
    friction_input[9] = x2e(0);
    friction_input[10] = x2e(1);
    friction_input[11] = x2e(2);
    friction_input[12] = x1s0(0);
    friction_input[13] = x1s0(1);
    friction_input[14] = x1s0(2);
    friction_input[15] = x1e0(0);
    friction_input[16] = x1e0(1);
    friction_input[17] = x1e0(2);
    friction_input[18] = x2s0(0);
    friction_input[19] = x2s0(1);
    friction_input[20] = x2s0(2);
    friction_input[21] = x2e0(0);
    friction_input[22] = x2e0(1);
    friction_input[23] = x2e0(2);
    friction_input[24] = contact_gradient(0);
    friction_input[25] = contact_gradient(1);
    friction_input[26] = contact_gradient(2);
    friction_input[27] = contact_gradient(3);
    friction_input[28] = contact_gradient(4);
    friction_input[29] = contact_gradient(5);
    friction_input[30] = contact_gradient(6);
    friction_input[31] = contact_gradient(7);
    friction_input[32] = contact_gradient(8);
    friction_input[33] = contact_gradient(9);
    friction_input[34] = contact_gradient(10);
    friction_input[35] = contact_gradient(11);
}


void contactPotentialIMC::computeFc(bool first_iter) {
    int edge1, edge2;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->candidate_ids(i, 0);
        edge2 = col_detector->candidate_ids(i, 1);
        prepContactInput(edge1, edge2);

        contact_potential_gradient_func.call(contact_gradient.data(), contact_input.data());

        contact_gradient *= contact_stiffness;

        if (friction && !first_iter) {
            prepFrictionInput(edge1, edge2);

            friction_force_func.call(friction_forces.data(), friction_input.data());

            contact_gradient += friction_forces;
        }

        // Add contact potential gradient and friction forces
        for (int e1 = 0; e1 < 6; e1++) {
            stepper->addForce(4*edge1+di[e1], contact_gradient[e1]);
        }
        for (int e2 = 0; e2 < 6; e2++) {
            stepper->addForce(4*edge2+di[e2], contact_gradient[e2+6]);
        }

    }
}


void contactPotentialIMC::computeFcJc(bool first_iter) {
    int edge1, edge2;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->candidate_ids(i, 0);
        edge2 = col_detector->candidate_ids(i, 1);
        prepContactInput(edge1, edge2);

        contact_potential_gradient_func.call(contact_gradient.data(), contact_input.data());
        contact_potential_hessian_func.call(contact_hessian.data(), contact_input.data());

        contact_gradient *= contact_stiffness;
        contact_hessian *= contact_stiffness;

        if (friction && !first_iter) {
            prepFrictionInput(edge1, edge2);

            friction_force_func.call(friction_forces.data(), friction_input.data());
            friction_partials_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
            friction_partials_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());

            // Chain ruling to get complete friction Jacobian
            friction_jacobian = friction_partials_dfr_dx + friction_partials_dfr_dfc.transpose() * contact_hessian;

            contact_gradient += friction_forces;
            contact_hessian += friction_jacobian;
        }

        // Add contact potential gradient and friction forces
        for (int e1 = 0; e1 < 6; e1++) {
            stepper->addForce(4*edge1+di[e1], contact_gradient[e1]);
        }
        for (int e2 = 0; e2 < 6; e2++) {
            stepper->addForce(4*edge2+di[e2], contact_gradient[e2+6]);
        }

        // Add contact potential Hessian and friction Jacobian
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 6; k++) {
                stepper->addJacobian(4*edge1+di[k], 4*edge1+di[j], contact_hessian(j, k));
            }
            for (int k = 6; k < 12; k++) {
                stepper->addJacobian(4*edge2+di[k-6], 4*edge1+di[j], contact_hessian(j, k));
            }
        }
        for (int j = 6; j < 12; j++) {
            for (int k = 0; k < 6; k++) {
                stepper->addJacobian(4*edge1+di[k], 4*edge2+di[j-6], contact_hessian(j, k));
            }
            for (int k = 6; k < 12; k++) {
                stepper->addJacobian(4*edge2+di[k-6], 4*edge2+di[j-6], contact_hessian(j, k));
            }
        }
    }
}


void contactPotentialIMC::generatePotentialFunctions() {
    auto x1s_x = symbol("x1s_x");
    auto x1s_y = symbol("x1s_y");
    auto x1s_z = symbol("x1s_z");
    auto x1e_x = symbol("x1e_x");
    auto x1e_y = symbol("x1e_y");
    auto x1e_z = symbol("x1e_z");
    auto x2s_x = symbol("x2s_x");
    auto x2s_y = symbol("x2s_y");
    auto x2s_z = symbol("x2s_z");
    auto x2e_x = symbol("x2e_x");
    auto x2e_y = symbol("x2e_y");
    auto x2e_z = symbol("x2e_z");

    vec_basic nodes_vec {x1s_x, x1s_y, x1s_z,
                         x1e_x, x1e_y, x1e_z,
                         x2s_x, x2s_y, x2s_z,
                         x2e_x, x2e_y, x2e_z};
    DenseMatrix nodes {nodes_vec};

    auto ce_k = symbol("ce_k");
    auto h2 = symbol("h2");

    vec_basic func_inputs(nodes_vec);
    func_inputs.push_back(ce_k);
    func_inputs.push_back(h2);

    // Construct Symbolic Arrays for each Node
    DenseMatrix x1s({x1s_x, x1s_y, x1s_z});
    DenseMatrix x1e({x1e_x, x1e_y, x1e_z});
    DenseMatrix x2s({x2s_x, x2s_y, x2s_z});
    DenseMatrix x2e({x2e_x, x2e_y, x2e_z});

    int num_rows = x1s.nrows();
    int num_cols = x1s.ncols();

    DenseMatrix e1(num_rows, num_cols);
    DenseMatrix e2(num_rows, num_cols);
    DenseMatrix e12(num_rows, num_cols);

    subtract_matrix(x1e, x1s, e1);
    subtract_matrix(x2e, x2s, e2);
    subtract_matrix(x2s, x1s, e12);

    DenseMatrix e1_squared(num_rows, num_cols);
    DenseMatrix e2_squared(num_rows, num_cols);
    DenseMatrix e1_e12(num_rows, num_cols);
    DenseMatrix e2_e12(num_rows, num_cols);
    DenseMatrix e1_e2(num_rows, num_cols);
    e1.elementwise_mul_matrix(e1, e1_squared);
    e2.elementwise_mul_matrix(e2, e2_squared);
    e1.elementwise_mul_matrix(e12, e1_e12);
    e2.elementwise_mul_matrix(e12, e2_e12);
    e1.elementwise_mul_matrix(e2, e1_e2);

    auto D1 = add(e1_squared.as_vec_basic());
    auto D2 = add(e2_squared.as_vec_basic());
    auto S1 = add(e1_e12.as_vec_basic());
    auto S2 = add(e2_e12.as_vec_basic());
    auto R = add(e1_e2.as_vec_basic());

    auto den = sub(mul(D1, D2), pow(R, 2));

    auto t1 = div(sub(mul(S1, D2), mul(S2, R)), den);

    RCP<const Basic> t2;
    approx_fixbound(t1, t2, 50.0);

    auto u1 = div(sub(mul(t2, R), S2), D2);

    RCP<const Basic> u2;
    approx_fixbound(u1, u2, 50.0);

    RCP<const Basic> conditional;
    approx_boxcar(u1, conditional, 50.0);

    auto left_cond = mul(sub(one, conditional), div(add(mul(u2, R), S1), D1));
    auto right_cond = mul(conditional, t2);
    auto t3 = add(left_cond, right_cond);

    RCP<const Basic> t4;
    approx_fixbound(t3, t4, 50.0);

    DenseMatrix c1(num_rows, num_cols);
    DenseMatrix c2(num_rows, num_cols);

    e1.mul_scalar(t4, c1);
    e2.mul_scalar(u2, c2);

    DenseMatrix dist_xyz(num_rows, num_cols);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);

    DenseMatrix dist_xyz_squared(num_rows, num_cols);
    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz_squared);

    RCP<const Basic> dist = pow(add(dist_xyz_squared.as_vec_basic()), 0.5);

    RCP<const Basic> E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));

    DenseMatrix contact_potential{{E}};

    DenseMatrix contact_potential_gradient(1, 12);

    jacobian(contact_potential, nodes, contact_potential_gradient);

    DenseMatrix contact_potential_hessian(12, 12);
    jacobian(contact_potential_gradient, nodes, contact_potential_hessian);

    // Common subexpression elimination (CSE) is extremely important for efficiency
    bool symbolic_cse = true;
    int opt_level = 3;

    contact_potential_gradient_func.init(func_inputs, contact_potential_gradient.as_vec_basic(), symbolic_cse, opt_level);
    contact_potential_hessian_func.init(func_inputs, contact_potential_hessian.as_vec_basic(), symbolic_cse, opt_level);

    if (friction) {
        auto x1s_x0 = symbol("x1s_x0");
        auto x1s_y0 = symbol("x1s_y0");
        auto x1s_z0 = symbol("x1s_z0");
        auto x1e_x0 = symbol("x1e_x0");
        auto x1e_y0 = symbol("x1e_y0");
        auto x1e_z0 = symbol("x1e_z0");
        auto x2s_x0 = symbol("x2s_x0");
        auto x2s_y0 = symbol("x2s_y0");
        auto x2s_z0 = symbol("x2s_z0");
        auto x2e_x0 = symbol("x2e_x0");
        auto x2e_y0 = symbol("x2e_y0");
        auto x2e_z0 = symbol("x2e_z0");
        auto f1s_x = symbol("f1s_x");
        auto f1s_y = symbol("f1s_y");
        auto f1s_z = symbol("f1s_z");
        auto f1e_x = symbol("f1e_x");
        auto f1e_y = symbol("f1e_y");
        auto f1e_z = symbol("f1e_z");
        auto f2s_x = symbol("f2s_x");
        auto f2s_y = symbol("f2s_y");
        auto f2s_z = symbol("f2s_z");
        auto f2e_x = symbol("f2e_x");
        auto f2e_y = symbol("f2e_y");
        auto f2e_z = symbol("f2e_z");
        auto mu = symbol("mu");
        auto dt = symbol("dt");
        auto vel_tol = symbol("vel_tol");

        // Construct Symbolic Arrays
        DenseMatrix x1s_0({x1s_x0, x1s_y0, x1s_z0});
        DenseMatrix x1e_0({x1e_x0, x1e_y0, x1e_z0});
        DenseMatrix x2s_0({x2s_x0, x2s_y0, x2s_z0});
        DenseMatrix x2e_0({x2e_x0, x2e_y0, x2e_z0});
        DenseMatrix f1s({f1s_x, f1s_y, f1s_z});
        DenseMatrix f1e({f1e_x, f1e_y, f1e_z});
        DenseMatrix f2s({f2s_x, f2s_y, f2s_z});
        DenseMatrix f2e({f2e_x, f2e_y, f2e_z});
        DenseMatrix f1(num_rows, num_cols);
        DenseMatrix f2(num_rows, num_cols);
        f1s.add_matrix(f1e, f1);
        f2s.add_matrix(f2e, f2);

        vec_basic ffr_input {x1s_x, x1s_y, x1s_z,
                             x1e_x, x1e_y, x1e_z,
                             x2s_x, x2s_y, x2s_z,
                             x2e_x, x2e_y, x2e_z,
                             x1s_x0, x1s_y0, x1s_z0,
                             x1e_x0, x1e_y0, x1e_z0,
                             x2s_x0, x2s_y0, x2s_z0,
                             x2e_x0, x2e_y0, x2e_z0,
                             f1s_x, f1s_y, f1s_z,
                             f1e_x, f1e_y, f1e_z,
                             f2s_x, f2s_y, f2s_z,
                             f2e_x, f2e_y, f2e_z,
                             mu, dt, vel_tol};

        vec_basic cforces {f1s_x, f1s_y, f1s_z,
                           f1e_x, f1e_y, f1e_z,
                           f2s_x, f2s_y, f2s_z,
                           f2e_x, f2e_y, f2e_z};

        DenseMatrix f1s_squared(num_rows, num_cols);
        DenseMatrix f1e_squared(num_rows, num_cols);
        DenseMatrix f2s_squared(num_rows, num_cols);
        DenseMatrix f2e_squared(num_rows, num_cols);
        DenseMatrix f1_squared(num_rows, num_cols);
        DenseMatrix f2_squared(num_rows, num_cols);
        f1s.elementwise_mul_matrix(f1s, f1s_squared);
        f1e.elementwise_mul_matrix(f1e, f1e_squared);
        f2s.elementwise_mul_matrix(f2s, f2s_squared);
        f2e.elementwise_mul_matrix(f2e, f2e_squared);
        f1.elementwise_mul_matrix(f1, f1_squared);
        f2.elementwise_mul_matrix(f2, f2_squared);

        RCP<const Basic> f1s_n = pow(add(f1s_squared.as_vec_basic()), 0.5);
        RCP<const Basic> f1e_n = pow(add(f1e_squared.as_vec_basic()), 0.5);
        RCP<const Basic> f2s_n = pow(add(f2s_squared.as_vec_basic()), 0.5);
        RCP<const Basic> f2e_n = pow(add(f2e_squared.as_vec_basic()), 0.5);
        RCP<const Basic> f1_n = pow(add(f1_squared.as_vec_basic()), 0.5);
        RCP<const Basic> f2_n = pow(add(f2_squared.as_vec_basic()), 0.5);

        RCP<const Basic> t1 = div(f1s_n, f1_n);
        RCP<const Basic> t2 = div(f1e_n, f1_n);
        RCP<const Basic> u1 = div(f2s_n, f2_n);
        RCP<const Basic> u2 = div(f2e_n, f2_n);

        DenseMatrix f1_nu(num_rows, num_cols);
        DenseMatrix f2_nu(num_rows, num_cols);
        f1.mul_scalar(div(one, f1_n), f1_nu);
        f2.mul_scalar(div(one, f2_n), f2_nu);

        DenseMatrix v1s(num_rows, num_cols);
        DenseMatrix v1e(num_rows, num_cols);
        DenseMatrix v2s(num_rows, num_cols);
        DenseMatrix v2e(num_rows, num_cols);
        subtract_matrix(x1s, x1s_0, v1s);
        subtract_matrix(x1e, x1e_0, v1e);
        subtract_matrix(x2s, x2s_0, v2s);
        subtract_matrix(x2e, x2e_0, v2e);

        DenseMatrix v1s_r(num_rows, num_cols);
        DenseMatrix v1e_r(num_rows, num_cols);
        DenseMatrix v2s_r(num_rows, num_cols);
        DenseMatrix v2e_r(num_rows, num_cols);
        v1s.mul_scalar(t1, v1s_r);
        v1e.mul_scalar(t2, v1e_r);
        v2s.mul_scalar(u1, v2s_r);
        v2e.mul_scalar(u2, v2e_r);

        DenseMatrix v1(num_rows, num_cols);
        DenseMatrix v2(num_rows, num_cols);
        v1s_r.add_matrix(v1e_r, v1);
        v2s_r.add_matrix(v2e_r, v2);

        DenseMatrix v_rel1(num_rows, num_cols);
        DenseMatrix v_rel2(num_rows, num_cols);
        subtract_matrix(v1, v2, v_rel1);
        subtract_matrix(v2, v1, v_rel2);

        // Compute tangent velocity of edge 1
        DenseMatrix tv_rel1_dot_vec(num_rows, num_cols);
        v_rel1.elementwise_mul_matrix(f1_nu, tv_rel1_dot_vec);
        RCP<const Basic> tv_rel1_dot = add(tv_rel1_dot_vec.as_vec_basic());
        DenseMatrix tv_rel1_component(num_rows, num_cols);
        f1_nu.mul_scalar(tv_rel1_dot, tv_rel1_component);
        DenseMatrix tv_rel1(num_rows, num_cols);
        subtract_matrix(v_rel1, tv_rel1_component, tv_rel1);
        DenseMatrix tv_rel1_squared(num_rows, num_cols);
        tv_rel1.elementwise_mul_matrix(tv_rel1, tv_rel1_squared);
        RCP<const Basic> tv_rel1_n = pow(add(tv_rel1_squared.as_vec_basic()), 0.5);
        DenseMatrix tv_rel1_u(num_rows, num_cols);
        tv_rel1.mul_scalar(div(one, tv_rel1_n), tv_rel1_u);

        // Compute tangent velocity of edge 2
        DenseMatrix tv_rel2_dot_vec(num_rows, num_cols);
        v_rel2.elementwise_mul_matrix(f2_nu, tv_rel2_dot_vec);
        RCP<const Basic> tv_rel2_dot = add(tv_rel2_dot_vec.as_vec_basic());
        DenseMatrix tv_rel2_component(num_rows, num_cols);
        f2_nu.mul_scalar(tv_rel2_dot, tv_rel2_component);
        DenseMatrix tv_rel2(num_rows, num_cols);
        subtract_matrix(v_rel2, tv_rel2_component, tv_rel2);
        DenseMatrix tv_rel2_squared(num_rows, num_cols);
        tv_rel2.elementwise_mul_matrix(tv_rel2, tv_rel2_squared);
        RCP<const Basic> tv_rel2_n = pow(add(tv_rel2_squared.as_vec_basic()), 0.5);
        DenseMatrix tv_rel2_u(num_rows, num_cols);
        tv_rel2.mul_scalar(div(one, tv_rel2_n), tv_rel2_u);

        RCP<const Basic> tv_rel1_n_scaled = mul(mul(div(one, dt), vel_tol), tv_rel1_n);
        RCP<const Basic> tv_rel2_n_scaled = mul(mul(div(one, dt), vel_tol), tv_rel2_n);

        RCP<const Basic> heaviside1 = sub(div(integer(2), add(one, exp(mul(integer(-1), tv_rel1_n_scaled)))), one);
        RCP<const Basic> heaviside2 = sub(div(integer(2), add(one, exp(mul(integer(-1), tv_rel2_n_scaled)))), one);

        RCP<const Basic> ffr1_scalar = mul(mul(heaviside1, mu), f1_n);
        RCP<const Basic> ffr2_scalar = mul(mul(heaviside2, mu), f2_n);

        DenseMatrix ffr1(num_rows, num_cols);
        DenseMatrix ffr2(num_rows, num_cols);
        tv_rel1_u.mul_scalar(ffr1_scalar, ffr1);
        tv_rel2_u.mul_scalar(ffr2_scalar, ffr2);

        DenseMatrix ffr1s(num_rows, num_cols);
        DenseMatrix ffr1e(num_rows, num_cols);
        DenseMatrix ffr2s(num_rows, num_cols);
        DenseMatrix ffr2e(num_rows, num_cols);
        ffr1.mul_scalar(t1, ffr1s);
        ffr1.mul_scalar(t2, ffr1e);
        ffr2.mul_scalar(u1, ffr2s);
        ffr2.mul_scalar(u2, ffr2e);

        DenseMatrix ffr_vec({ffr1s.get(0, 0), ffr1s.get(1, 0), ffr1s.get(2, 0),
                             ffr1e.get(0, 0), ffr1e.get(1, 0), ffr1e.get(2, 0),
                             ffr2s.get(0, 0), ffr2s.get(1, 0), ffr2s.get(2, 0),
                             ffr2e.get(0, 0), ffr2e.get(1, 0), ffr2e.get(2, 0)});

        DenseMatrix friction_partial_dfr_dx(12, 12);
        DenseMatrix friction_partial_dfr_dfc(12, 12);
        jacobian(ffr_vec, nodes, friction_partial_dfr_dx);
        jacobian(ffr_vec, cforces, friction_partial_dfr_dfc);

        friction_force_func.init(ffr_input, ffr_vec.as_vec_basic(), symbolic_cse, opt_level);
        friction_partials_dfr_dx_func.init(ffr_input, friction_partial_dfr_dx.as_vec_basic(), symbolic_cse, opt_level);
        friction_partials_dfr_dfc_func.init(ffr_input, friction_partial_dfr_dfc.as_vec_basic(), symbolic_cse, opt_level);
    };
}
