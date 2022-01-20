#include "contactPotentialIMC.h"


contactPotentialIMC::contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, collisionDetector &m_col_detector,
                                         int m_ce_k, int m_friction, double m_mu, double m_vel_tol) {
    rod = &m_rod;
    stepper = &m_stepper;
    col_detector = &m_col_detector;

    ce_k = m_ce_k;
    scale = 1 / rod->rodRadius;
    h2 = scale * rod->rodRadius * 2;
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

    sym_eqs = new symbolicEquations();
    sym_eqs->generateContactPotentialFunctions();
    sym_eqs->generateContactPotentialFunctionsT0();
    sym_eqs->generateContactPotentialFunctionsT1();
    sym_eqs->generateParallelContactPotentialFunctions();
    if (friction) {
        sym_eqs->generateFrictionJacobianFunctions();
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
    Vector3d x1s = scale * rod->getVertex(edge1);
    Vector3d x1e = scale * rod->getVertex(edge1+1);
    Vector3d x2s = scale * rod->getVertex(edge2);
    Vector3d x2e = scale * rod->getVertex(edge2+1);

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


void contactPotentialIMC::computeFriction(int edge1, int edge2) {
    Vector3d x1s = rod->getVertex(edge1);
    Vector3d x1e = rod->getVertex(edge1+1);
    Vector3d x2s = rod->getVertex(edge2);
    Vector3d x2e = rod->getVertex(edge2+1);
    Vector3d x1s0 = rod->getPreVertex(edge1);
    Vector3d x1e0 = rod->getPreVertex(edge1+1);
    Vector3d x2s0 = rod->getPreVertex(edge2);
    Vector3d x2e0 = rod->getPreVertex(edge2+1);
    Vector3d f1s = contact_gradient(seq(0, 2));
    Vector3d f1e = contact_gradient(seq(3, 5));
    Vector3d f2s = contact_gradient(seq(6, 8));
    Vector3d f2e = contact_gradient(seq(9, 11));

    double f1s_n = f1s.norm();
    double f2s_n = f2s.norm();

    double fn = (f1s + f1e).norm();

    double ffr_val = fn * mu;

    double t1 = f1s_n / fn;
    double u1 = f2s_n / fn;

    if (t1 > 1) t1 = 1;
    if (t1 < 0) t1 = 0;
    if (u1 > 1) u1 = 1;
    if (u1 < 0) u1 = 0;

    double t2 = 1 - t1;
    double u2 = 1 - u1;

    Vector3d v1s = x1s - x1s0;
    Vector3d v1e = x1e - x1e0;
    Vector3d v2s = x2s - x2s0;
    Vector3d v2e = x2e - x2e0;

    Vector3d v1 = t1 * v1s + t2 * v1e;
    Vector3d v2 = u1 * v2s + u2 * v2e;
    Vector3d v_rel = v1 - v2;

    Vector3d norm = (f1s + f1e) / fn;
    Vector3d tv_rel = v_rel - v_rel.dot(norm) * norm;
    double tv_rel_n = tv_rel.norm();
    Vector3d tv_rel_u = tv_rel / tv_rel_n;

    double tv_rel_n_scaled = tv_rel_n * (1 / rod->dt) * vel_tol;
    double heaviside = (2 / (1 + exp(-tv_rel_n_scaled))) - 1;

    Vector3d ffr_e = heaviside * tv_rel_u * ffr_val;

    friction_forces(seq(0, 2)) = t1 * ffr_e;
    friction_forces(seq(3, 5)) = t2 * ffr_e;
    friction_forces(seq(6, 8)) = -u1 * ffr_e;
    friction_forces(seq(9, 11)) = -u2 * ffr_e;
}


void contactPotentialIMC::computeFc(bool first_iter) {
    ParallelCase curr_case;
    double t_ref;
    int edge1, edge2;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->candidate_ids(i, 0);
        edge2 = col_detector->candidate_ids(i, 1);
        prepContactInput(edge1, edge2);

        curr_case = col_detector->parallel_cases[i];

        if (curr_case == NOPA) {
            col_detector->getTRefVal(edge1, edge2, t_ref);
            if (t_ref < -0.1) {
                sym_eqs->contact_potential_t0_grad_func.call(contact_gradient.data(), contact_input.data());
            }
            else if (t_ref > 1.1) {
                sym_eqs->contact_potential_t1_grad_func.call(contact_gradient.data(), contact_input.data());
            }
            else {
                sym_eqs->contact_potential_grad_func.call(contact_gradient.data(), contact_input.data());
            }
        }
        else if (curr_case == ACBD) {
            sym_eqs->parallel_ACBD_case_grad_func.call(contact_gradient.data(), contact_input.data());
        }
        else if (curr_case == ADBC) {
            sym_eqs->parallel_ADBC_case_grad_func.call(contact_gradient.data(), contact_input.data());
        }
        else if (curr_case == CADB) {
            sym_eqs->parallel_CADB_case_grad_func.call(contact_gradient.data(), contact_input.data());
        }
        else if (curr_case == DACB) {
            sym_eqs->parallel_DACB_case_grad_func.call(contact_gradient.data(), contact_input.data());
        }

        contact_gradient *= contact_stiffness;

        if (friction && !first_iter) {
            prepFrictionInput(edge1, edge2);

            computeFriction(edge1, edge2);

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
    ParallelCase curr_case;
    double t_ref;
    int edge1, edge2;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->candidate_ids(i, 0);
        edge2 = col_detector->candidate_ids(i, 1);
        prepContactInput(edge1, edge2);

        curr_case = col_detector->parallel_cases[i];

        if (curr_case == NOPA) {
            col_detector->getTRefVal(edge1, edge2, t_ref);
            if (t_ref < -0.1) {
                sym_eqs->contact_potential_t0_grad_func.call(contact_gradient.data(), contact_input.data());
                sym_eqs->contact_potential_t0_hess_func.call(contact_hessian.data(), contact_input.data());
            }
            else if (t_ref > 1.1) {
                sym_eqs->contact_potential_t1_grad_func.call(contact_gradient.data(), contact_input.data());
                sym_eqs->contact_potential_t1_hess_func.call(contact_hessian.data(), contact_input.data());
            }
            else {
                sym_eqs->contact_potential_grad_func.call(contact_gradient.data(), contact_input.data());
                sym_eqs->contact_potential_hess_func.call(contact_hessian.data(), contact_input.data());
            }
        }
        else if (curr_case == ACBD) {
            sym_eqs->parallel_ACBD_case_grad_func.call(contact_gradient.data(), contact_input.data());
            sym_eqs->parallel_ACBD_case_hess_func.call(contact_hessian.data(), contact_input.data());
        }
        else if (curr_case == ADBC) {
            sym_eqs->parallel_ADBC_case_grad_func.call(contact_gradient.data(), contact_input.data());
            sym_eqs->parallel_ADBC_case_hess_func.call(contact_hessian.data(), contact_input.data());
        }
        else if (curr_case == CADB) {
            sym_eqs->parallel_CADB_case_grad_func.call(contact_gradient.data(), contact_input.data());
            sym_eqs->parallel_CADB_case_hess_func.call(contact_hessian.data(), contact_input.data());
        }
        else if (curr_case == DACB) {
            sym_eqs->parallel_DACB_case_grad_func.call(contact_gradient.data(), contact_input.data());
            sym_eqs->parallel_DACB_case_hess_func.call(contact_hessian.data(), contact_input.data());
        }

        contact_gradient *= contact_stiffness;
        contact_hessian *= contact_stiffness * scale;

        if (friction && !first_iter) {
            prepFrictionInput(edge1, edge2);

            computeFriction(edge1, edge2);

            sym_eqs->friction_partials_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
            sym_eqs->friction_partials_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());

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
