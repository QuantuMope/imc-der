#include "symbolicEquations.h"


symbolicEquations::symbolicEquations() {
    x1s_x = symbol("x1s_x");
    x1s_y = symbol("x1s_y");
    x1s_z = symbol("x1s_z");
    x1e_x = symbol("x1e_x");
    x1e_y = symbol("x1e_y");
    x1e_z = symbol("x1e_z");
    x2s_x = symbol("x2s_x");
    x2s_y = symbol("x2s_y");
    x2s_z = symbol("x2s_z");
    x2e_x = symbol("x2e_x");
    x2e_y = symbol("x2e_y");
    x2e_z = symbol("x2e_z");
    ce_k = symbol("ce_k");
    h2 = symbol("h2");

    x1s_x0 = symbol("x1s_x0");
    x1s_y0 = symbol("x1s_y0");
    x1s_z0 = symbol("x1s_z0");
    x1e_x0 = symbol("x1e_x0");
    x1e_y0 = symbol("x1e_y0");
    x1e_z0 = symbol("x1e_z0");
    x2s_x0 = symbol("x2s_x0");
    x2s_y0 = symbol("x2s_y0");
    x2s_z0 = symbol("x2s_z0");
    x2e_x0 = symbol("x2e_x0");
    x2e_y0 = symbol("x2e_y0");
    x2e_z0 = symbol("x2e_z0");
    f1s_x = symbol("f1s_x");
    f1s_y = symbol("f1s_y");
    f1s_z = symbol("f1s_z");
    f1e_x = symbol("f1e_x");
    f1e_y = symbol("f1e_y");
    f1e_z = symbol("f1e_z");
    f2s_x = symbol("f2s_x");
    f2s_y = symbol("f2s_y");
    f2s_z = symbol("f2s_z");
    f2e_x = symbol("f2e_x");
    f2e_y = symbol("f2e_y");
    f2e_z = symbol("f2e_z");
    mu = symbol("mu");
    dt = symbol("dt");
    vel_tol = symbol("vel_tol");

    symbolic_cse = true;
    opt_level = 3;
}


void symbolicEquations::approx_fixbound(const RCP<const Basic> &input, RCP<const Basic> &result, const double &m_k) {
    const RCP<const RealDouble> k = real_double(m_k);
    auto first_term = log(add(one, exp(mul(k, input))));
    auto second_term = log(add(one, exp(mul(k, sub(input, one)))));
    result = mul(div(one, k), sub(first_term, second_term));
}


void symbolicEquations::approx_boxcar(const RCP<const Basic> &input, RCP<const Basic> &result, const double &m_k) {
    const RCP<const RealDouble> k = real_double(m_k);
    auto first_term = div(one, add(one, exp(mul(mul(integer(-1), k), input))));
    auto second_term = div(one, add(one, exp(mul(mul(integer(-1), k), sub(input, one)))));
    result = sub(first_term, second_term);
}


// For some reason SymEngine doesn't have this implemented X_X
void symbolicEquations::subtract_matrix(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
    assert((A.nrows() == B.nrows()) && (A.ncols() == B.ncols()));
    for (unsigned i=0; i < A.nrows(); i++) {
        for (unsigned j=0; j < A.ncols(); j++) {
            C.set(i, j, sub(A.get(i, j), B.get(i, j)));
        }
    }
}


void symbolicEquations::generateContactPotentialFunctions() {
    DenseMatrix nodes{{x1s_x, x1s_y, x1s_z,
                       x1e_x, x1e_y, x1e_z,
                       x2s_x, x2s_y, x2s_z,
                       x2e_x, x2e_y, x2e_z}};

    vec_basic func_inputs(nodes.as_vec_basic());
    func_inputs.push_back(ce_k);
    func_inputs.push_back(h2);

    // Construct Symbolic Arrays for each Nodes
    DenseMatrix x1s({x1s_x, x1s_y, x1s_z});
    DenseMatrix x1e({x1e_x, x1e_y, x1e_z});
    DenseMatrix x2s({x2s_x, x2s_y, x2s_z});
    DenseMatrix x2e({x2e_x, x2e_y, x2e_z});

    DenseMatrix e1(3, 1);
    DenseMatrix e2(3, 1);
    DenseMatrix e12(3, 1);

    subtract_matrix(x1e, x1s, e1);
    subtract_matrix(x2e, x2s, e2);
    subtract_matrix(x2s, x1s, e12);

    DenseMatrix e1_squared(3, 1);
    DenseMatrix e2_squared(3, 1);
    DenseMatrix e1_e12(3, 1);
    DenseMatrix e2_e12(3, 1);
    DenseMatrix e1_e2(3, 1);
    e1.elementwise_mul_matrix(e1, e1_squared);
    e2.elementwise_mul_matrix(e2, e2_squared);
    e1.elementwise_mul_matrix(e12, e1_e12);
    e2.elementwise_mul_matrix(e12, e2_e12);
    e1.elementwise_mul_matrix(e2, e1_e2);

    RCP<const Basic> D1 = add(e1_squared.as_vec_basic());
    RCP<const Basic> D2 = add(e2_squared.as_vec_basic());
    RCP<const Basic> S1 = add(e1_e12.as_vec_basic());
    RCP<const Basic> S2 = add(e2_e12.as_vec_basic());
    RCP<const Basic> R = add(e1_e2.as_vec_basic());

    RCP<const Basic> den = sub(mul(D1, D2), pow(R, 2));

    RCP<const Basic> t = div(sub(mul(S1, D2), mul(S2, R)), den);

    approx_fixbound(t, t, 50.0);

    RCP<const Basic> u = div(sub(mul(t, R), S2), D2);
    RCP<const Basic> uf;

    approx_fixbound(u, uf, 50.0);

    RCP<const Basic> conditional;
    approx_boxcar(u, conditional, 50.0);

    RCP<const Basic> cond = div(add(mul(uf, R), S1), D1);
    approx_fixbound(cond, cond, 50.0);
    t = add(mul(sub(one, conditional), cond), mul(conditional, t));

    DenseMatrix c1(3, 1);
    DenseMatrix c2(3, 1);

    e1.mul_scalar(t, c1);
    e2.mul_scalar(uf, c2);

    DenseMatrix dist_xyz(3, 1);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);

    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz);

    RCP<const Basic> dist = pow(add(dist_xyz.as_vec_basic()), 0.5);

    RCP<const Basic> E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));

    DenseMatrix contact_potential{{E}};

    DenseMatrix contact_potential_gradient(1, 12);
    jacobian(contact_potential, nodes, contact_potential_gradient);

    DenseMatrix contact_potential_hessian(12, 12);
    jacobian(contact_potential_gradient, nodes, contact_potential_hessian);

    contact_potential_grad_func.init(func_inputs, contact_potential_gradient.as_vec_basic(), symbolic_cse, opt_level);
    contact_potential_hess_func.init(func_inputs, contact_potential_hessian.as_vec_basic(), symbolic_cse, opt_level);
}


void symbolicEquations::generateFrictionJacobianFunctions() {
    DenseMatrix nodes{{x1s_x, x1s_y, x1s_z,
                       x1e_x, x1e_y, x1e_z,
                       x2s_x, x2s_y, x2s_z,
                       x2e_x, x2e_y, x2e_z}};

    // Construct Symbolic Arrays for each Nodes
    DenseMatrix x1s({x1s_x, x1s_y, x1s_z});
    DenseMatrix x1e({x1e_x, x1e_y, x1e_z});
    DenseMatrix x2s({x2s_x, x2s_y, x2s_z});
    DenseMatrix x2e({x2e_x, x2e_y, x2e_z});
    DenseMatrix x1s_0({x1s_x0, x1s_y0, x1s_z0});
    DenseMatrix x1e_0({x1e_x0, x1e_y0, x1e_z0});
    DenseMatrix x2s_0({x2s_x0, x2s_y0, x2s_z0});
    DenseMatrix x2e_0({x2e_x0, x2e_y0, x2e_z0});
    DenseMatrix f1s({f1s_x, f1s_y, f1s_z});
    DenseMatrix f1e({f1e_x, f1e_y, f1e_z});
    DenseMatrix f2s({f2s_x, f2s_y, f2s_z});
    DenseMatrix f2e({f2e_x, f2e_y, f2e_z});
    DenseMatrix f1(3, 1);
    DenseMatrix f2(3, 1);
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

    DenseMatrix f1s_squared(3, 1);
    DenseMatrix f1e_squared(3, 1);
    DenseMatrix f2s_squared(3, 1);
    DenseMatrix f2e_squared(3, 1);
    DenseMatrix f1_squared(3, 1);
    DenseMatrix f2_squared(3, 1);
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

    DenseMatrix f1_nu(3, 1);
    DenseMatrix f2_nu(3, 1);
    f1.mul_scalar(div(one, f1_n), f1_nu);
    f2.mul_scalar(div(one, f2_n), f2_nu);

    DenseMatrix v1s(3, 1);
    DenseMatrix v1e(3, 1);
    DenseMatrix v2s(3, 1);
    DenseMatrix v2e(3, 1);
    subtract_matrix(x1s, x1s_0, v1s);
    subtract_matrix(x1e, x1e_0, v1e);
    subtract_matrix(x2s, x2s_0, v2s);
    subtract_matrix(x2e, x2e_0, v2e);

    DenseMatrix v1s_r(3, 1);
    DenseMatrix v1e_r(3, 1);
    DenseMatrix v2s_r(3, 1);
    DenseMatrix v2e_r(3, 1);
    v1s.mul_scalar(t1, v1s_r);
    v1e.mul_scalar(t2, v1e_r);
    v2s.mul_scalar(u1, v2s_r);
    v2e.mul_scalar(u2, v2e_r);

    DenseMatrix v1(3, 1);
    DenseMatrix v2(3, 1);
    v1s_r.add_matrix(v1e_r, v1);
    v2s_r.add_matrix(v2e_r, v2);

    DenseMatrix v_rel(3, 1);
    subtract_matrix(v1, v2, v_rel);

    // Compute tangent velocity of edge 1
    DenseMatrix tv_rel(3, 1);
    v_rel.elementwise_mul_matrix(f1_nu, tv_rel);
    RCP<const Basic> tmp = add(tv_rel.as_vec_basic());
    f1_nu.mul_scalar(tmp, tv_rel);
    subtract_matrix(v_rel, tv_rel, tv_rel);

    DenseMatrix tv_rel_squared(3, 1);
    tv_rel.elementwise_mul_matrix(tv_rel, tv_rel_squared);

    RCP<const Basic> tv_rel_n = pow(add(tv_rel_squared.as_vec_basic()), 0.5);

    DenseMatrix tv_rel_u(3, 1);
    tv_rel.mul_scalar(div(one, tv_rel_n), tv_rel_u);

    RCP<const Basic> tv_rel_n_scaled = mul(mul(div(one, dt), vel_tol), tv_rel_n);

    RCP<const Basic> heaviside = sub(div(integer(2), add(one, exp(mul(integer(-1), tv_rel_n_scaled)))), one);

    RCP<const Basic> ffr_scalar = mul(mul(heaviside, mu), f1_n);

    DenseMatrix ffr1(3, 1);
    DenseMatrix ffr2(3, 1);
    tv_rel_u.mul_scalar(ffr_scalar, ffr1);
    ffr1.mul_scalar(integer(-1), ffr2);

    DenseMatrix ffr1s(3, 1);
    DenseMatrix ffr1e(3, 1);
    DenseMatrix ffr2s(3, 1);
    DenseMatrix ffr2e(3, 1);
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

    friction_partials_dfr_dx_func.init(ffr_input, friction_partial_dfr_dx.as_vec_basic(), symbolic_cse, opt_level);
    friction_partials_dfr_dfc_func.init(ffr_input, friction_partial_dfr_dfc.as_vec_basic(), symbolic_cse, opt_level);
}


void symbolicEquations::generateParallelContactPotentialFunctions() {
    DenseMatrix nodes{{x1s_x, x1s_y, x1s_z,
                       x1e_x, x1e_y, x1e_z,
                       x2s_x, x2s_y, x2s_z,
                       x2e_x, x2e_y, x2e_z}};

    vec_basic func_inputs(nodes.as_vec_basic());
    func_inputs.push_back(ce_k);
    func_inputs.push_back(h2);

    // Construct Symbolic Arrays for each Node
    DenseMatrix x1s({x1s_x, x1s_y, x1s_z});
    DenseMatrix x1e({x1e_x, x1e_y, x1e_z});
    DenseMatrix x2s({x2s_x, x2s_y, x2s_z});
    DenseMatrix x2e({x2e_x, x2e_y, x2e_z});

    DenseMatrix e1(3, 1);
    DenseMatrix e2(3, 1);
    DenseMatrix e12(3, 1);

    subtract_matrix(x1e, x1s, e1);
    subtract_matrix(x2e, x2s, e2);
    subtract_matrix(x2s, x1s, e12);

    DenseMatrix e1_squared(3, 1);
    DenseMatrix e2_squared(3, 1);
    DenseMatrix e1_e12(3, 1);
    DenseMatrix e2_e12(3, 1);
    DenseMatrix e1_e2(3, 1);
    e1.elementwise_mul_matrix(e1, e1_squared);
    e2.elementwise_mul_matrix(e2, e2_squared);
    e1.elementwise_mul_matrix(e12, e1_e12);
    e2.elementwise_mul_matrix(e12, e2_e12);
    e1.elementwise_mul_matrix(e2, e1_e2);

    DenseMatrix e1_u(3, 1);
    RCP<const Basic> e1_n = pow(add(e1_squared.as_vec_basic()), 0.5);
    e1.mul_scalar(div(one, e1_n), e1_u);
    DenseMatrix axis({one, zero, zero});

    DenseMatrix v(3, 1);
    cross(e1_u, axis, v);
    DenseMatrix tmp_scalar(1, 1);
    dot(e1_u, axis, tmp_scalar);
    RCP<const Basic> c = tmp_scalar.get(0, 0);
    DenseMatrix v_squared(3, 1);
    v.elementwise_mul_matrix(v, v_squared);
    RCP<const Basic> s = pow(add(v_squared.as_vec_basic()), 0.5);

    DenseMatrix kmat({zero, mul(integer(-1), v.get(2, 0)), v.get(1, 0),
                      v.get(2, 0), zero, mul(integer(-1), v.get(0, 0)),
                      mul(integer(-1), v.get(1, 0)), v.get(0, 0), zero});
    kmat.resize(3, 3);

    DenseMatrix eye({one, zero, zero,
                     zero, one, zero,
                     zero, zero, one});
    eye.resize(3, 3);

    DenseMatrix mat(3, 3);
    kmat.mul_matrix(kmat, mat);
    mat.mul_scalar(div(sub(one, c), pow(s, 2)), mat);
    mat.add_matrix(kmat, mat);
    mat.add_matrix(eye, mat);

    // Just get the first row of rotation matrix
    DenseMatrix mat_x({mat.get(0, 0), mat.get(0, 1), mat.get(0, 2)});

    DenseMatrix x2s_p(3, 1);
    DenseMatrix x2e_p(3, 1);
    subtract_matrix(x2s, x1s, x2s_p);
    subtract_matrix(x2e, x1s, x2e_p);

    RCP<const Basic> B = e1_n;
    dot(mat_x, x2s_p, tmp_scalar);
    RCP<const Basic> C = tmp_scalar.get(0, 0);
    dot(mat_x, x2e_p, tmp_scalar);
    RCP<const Basic> D = tmp_scalar.get(0, 0);

    // Four cases for parallel edge contact
    RCP<const Basic> t, u;
    DenseMatrix c1(3, 1);
    DenseMatrix c2(3, 1);
    DenseMatrix dist_xyz(3, 1);
    RCP<const Basic> dist, E;

    // A < C < B < D
    t = div(div(add(B, C), integer(2)), B);
    u = div(div(sub(B, C), integer(2)), sub(D, C));
    e1.mul_scalar(t, c1);
    e2.mul_scalar(u, c2);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);
    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz);
    dist = pow(add(dist_xyz.as_vec_basic()), 0.5);
    E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));
    DenseMatrix ACBD{{E}};
    DenseMatrix ACBD_gradient(1, 12);
    jacobian(ACBD, nodes, ACBD_gradient);
    DenseMatrix ACBD_hessian(12, 12);
    jacobian(ACBD_gradient, nodes, ACBD_hessian);
    parallel_ACBD_case_grad_func.init(func_inputs, ACBD_gradient.as_vec_basic(), symbolic_cse, opt_level);
    parallel_ACBD_case_hess_func.init(func_inputs, ACBD_hessian.as_vec_basic(), symbolic_cse, opt_level);

    // A < D < B < C
    t = div(div(add(B, D), integer(2)), B);
    u = sub(one, div(div(sub(B, D), integer(2)), sub(C, D)));
    e1.mul_scalar(t, c1);
    e2.mul_scalar(u, c2);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);
    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz);
    dist = pow(add(dist_xyz.as_vec_basic()), 0.5);
    E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));
    DenseMatrix ADBC{{E}};
    DenseMatrix ADBC_gradient(1, 12);
    jacobian(ADBC, nodes, ADBC_gradient);
    DenseMatrix ADBC_hessian(12, 12);
    jacobian(ADBC_gradient, nodes, ADBC_hessian);
    parallel_ADBC_case_grad_func.init(func_inputs, ADBC_gradient.as_vec_basic(), symbolic_cse, opt_level);
    parallel_ADBC_case_hess_func.init(func_inputs, ADBC_hessian.as_vec_basic(), symbolic_cse, opt_level);

    // C < A < D < B
    t = div(div(D, B), integer(2));
    u = div(sub(div(D, integer(2)), C), sub(D, C));
    e1.mul_scalar(t, c1);
    e2.mul_scalar(u, c2);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);
    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz);
    dist = pow(add(dist_xyz.as_vec_basic()), 0.5);
    E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));
    DenseMatrix CADB{{E}};
    DenseMatrix CADB_gradient(1, 12);
    jacobian(CADB, nodes, CADB_gradient);
    DenseMatrix CADB_hessian(12, 12);
    jacobian(CADB_gradient, nodes, CADB_hessian);
    parallel_CADB_case_grad_func.init(func_inputs, CADB_gradient.as_vec_basic(), symbolic_cse, opt_level);
    parallel_CADB_case_hess_func.init(func_inputs, CADB_hessian.as_vec_basic(), symbolic_cse, opt_level);

    // D < A < C < B
    t = div(div(C, B), integer(2));
    u = sub(one, div(sub(div(C, integer(2)), D), sub(C, D)));
    e1.mul_scalar(t, c1);
    e2.mul_scalar(u, c2);
    subtract_matrix(c1, c2, dist_xyz);
    subtract_matrix(dist_xyz, e12, dist_xyz);
    dist_xyz.elementwise_mul_matrix(dist_xyz, dist_xyz);
    dist = pow(add(dist_xyz.as_vec_basic()), 0.5);
    E = mul(div(one, ce_k), log(add(one, exp(mul(ce_k, sub(h2, dist))))));
    DenseMatrix DACB{{E}};
    DenseMatrix DACB_gradient(1, 12);
    jacobian(DACB, nodes, DACB_gradient);
    DenseMatrix DACB_hessian(12, 12);
    jacobian(DACB_gradient, nodes, DACB_hessian);
    parallel_DACB_case_grad_func.init(func_inputs, DACB_gradient.as_vec_basic(), symbolic_cse, opt_level);
    parallel_DACB_case_hess_func.init(func_inputs, DACB_hessian.as_vec_basic(), symbolic_cse, opt_level);
}
