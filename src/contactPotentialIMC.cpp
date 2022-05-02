#include "contactPotentialIMC.h"


contactPotentialIMC::contactPotentialIMC(elasticRod &m_rod, timeStepper &m_stepper, collisionDetector &m_col_detector,
                                         double m_delta, double m_mu, double m_nu) {
    rod = &m_rod;
    stepper = &m_stepper;
    col_detector = &m_col_detector;

    delta = m_delta;
    mu = m_mu;
    nu = m_nu;
    friction = mu > 0.0;

    scale = 1 / rod->rodRadius;
    h2 = rod->rodRadius * 2;

    K1 = (15 * rod->rodRadius) / delta;
    K2 = 15 / nu;

    // Setup constant inputs
    p2p_input[6] = K1;
    p2p_input[7] = h2 * scale;

    e2p_input[9] = K1;
    e2p_input[10] = h2 * scale;

    e2e_input[12] = K1;
    e2e_input[13] = h2 * scale;

    friction_input[36] = mu;
    friction_input[37] = rod->dt;
    friction_input[38] = K2;

    sym_eqs = new symbolicEquations();
    sym_eqs->generateContactPotentialPiecewiseFunctions1();
    sym_eqs->generateContactPotentialPiecewiseFunctions2();
    if (friction) {
        sym_eqs->generateFrictionJacobianPiecewiseFunctions1();
        sym_eqs->generateFrictionJacobianPiecewiseFunctions2();
    }
}


void contactPotentialIMC::updateContactStiffness() {
    if (col_detector->candidateSet.size() == 0) return;
    double curr_max_force = -1;
    double curr_force;
    double fx, fy, fz;
    set<int> nodes_to_check;

    // Compute the maximum force that a node experiences.
    for (int i = 0; i < col_detector->candidateSet.size(); i++) {
        nodes_to_check.insert(col_detector->candidateSet[i][0]);
        nodes_to_check.insert(col_detector->candidateSet[i][0]+1);
        nodes_to_check.insert(col_detector->candidateSet[i][1]);
        nodes_to_check.insert(col_detector->candidateSet[i][1]+1);
    }

    for (auto i : nodes_to_check) {
        if (rod->getIfConstrained(4*i) == 0 &&
            rod->getIfConstrained(4*i+1) == 0 &&
            rod->getIfConstrained(4*i+2) == 0) {
            fx = stepper->getForce()[rod->fullToUnconsMap[4*i]];
            fy = stepper->getForce()[rod->fullToUnconsMap[4*i+1]];
            fz = stepper->getForce()[rod->fullToUnconsMap[4*i+2]];
        }
        else {
            continue;
        }
        curr_force = sqrt(pow(fx, 2) + pow(fy, 2) + pow(fz, 2));
        if (curr_force > curr_max_force) {
            curr_max_force = curr_force;
        }
    }
//    contact_stiffness = 1e5 * curr_max_force;
    contact_stiffness = 200;
}


void contactPotentialIMC::prepContactInput(int edge1, int edge2, int edge3, int edge4, int constraintType) {
    Vector3d x1s = scale * rod->getVertex(edge1);
    Vector3d x1e = scale * rod->getVertex(edge3);
    Vector3d x2s = scale * rod->getVertex(edge2);
    Vector3d x2e = scale * rod->getVertex(edge4);

    if (constraintType == 0) //p2p
    {
        p2p_input[0] = x1s(0);
        p2p_input[1] = x1s(1);
        p2p_input[2] = x1s(2);

        p2p_input[3] = x2s(0);
        p2p_input[4] = x2s(1);
        p2p_input[5] = x2s(2);
    }
    else if (constraintType == 1) //e2p
    {
        e2p_input[0] = x1s(0);
        e2p_input[1] = x1s(1);
        e2p_input[2] = x1s(2);
        e2p_input[3] = x1e(0);
        e2p_input[4] = x1e(1);
        e2p_input[5] = x1e(2);
        e2p_input[6] = x2s(0);
        e2p_input[7] = x2s(1);
        e2p_input[8] = x2s(2);
    }
    else if (constraintType == 2) //e2e
    {
        e2e_input[0] = x1s(0);
        e2e_input[1] = x1s(1);
        e2e_input[2] = x1s(2);
        e2e_input[3] = x1e(0);
        e2e_input[4] = x1e(1);
        e2e_input[5] = x1e(2);
        e2e_input[6] = x2s(0);
        e2e_input[7] = x2s(1);
        e2e_input[8] = x2s(2);
        e2e_input[9] = x2e(0);
        e2e_input[10] = x2e(1);
        e2e_input[11] = x2e(2);
    }
}


void contactPotentialIMC::prepFrictionInput(const int edge1, const int edge2, const int edge3, const int edge4) {
    Vector3d x1s = rod->getVertex(edge1);
    Vector3d x1e = rod->getVertex(edge3);
    Vector3d x2s = rod->getVertex(edge2);
    Vector3d x2e = rod->getVertex(edge4);
    Vector3d x1s0 = rod->getPreVertex(edge1);
    Vector3d x1e0 = rod->getPreVertex(edge3);
    Vector3d x2s0 = rod->getPreVertex(edge2);
    Vector3d x2e0 = rod->getPreVertex(edge4);

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


void contactPotentialIMC::computeFriction(const int edge1, const int edge2, const int edge3, const int edge4) {
    Vector3d x1s = rod->getVertex(edge1);
    Vector3d x1e = rod->getVertex(edge3);
    Vector3d x2s = rod->getVertex(edge2);
    Vector3d x2e = rod->getVertex(edge4);
    Vector3d x1s0 = rod->getPreVertex(edge1);
    Vector3d x1e0 = rod->getPreVertex(edge3);
    Vector3d x2s0 = rod->getPreVertex(edge2);
    Vector3d x2e0 = rod->getPreVertex(edge4);
    Vector3d f1s = contact_gradient(seq(0, 2));
    Vector3d f1e = contact_gradient(seq(3, 5));
    Vector3d f2s = contact_gradient(seq(6, 8));
    Vector3d f2e = contact_gradient(seq(9, 11));

    double f1s_n = f1s.norm();
    double f1e_n = f1e.norm();
    double f2s_n = f2s.norm();
    double f2e_n = f2e.norm();

    double fn = (f1s + f1e).norm();

    double beta11 = f1s_n / fn;
    double beta21 = f2s_n / fn;

    if (beta11 > 1) beta11 = 1;
    if (beta11 < 0) beta11 = 0;
    if (beta21 > 1) beta21 = 1;
    if (beta21 < 0) beta21 = 0;

    double beta12 = 1 - beta11;
    double beta22 = 1 - beta21;

    Vector3d v1s = (x1s - x1s0) / rod->dt;
    Vector3d v1e = (x1e - x1e0) / rod->dt;
    Vector3d v2s = (x2s - x2s0) / rod->dt;
    Vector3d v2e = (x2e - x2e0) / rod->dt;

    Vector3d v1 = beta11 * v1s + beta12 * v1e;
    Vector3d v2 = beta21 * v2s + beta22 * v2e;
    Vector3d v_rel = v1 - v2;

    Vector3d contact_norm = (f1s + f1e) / fn;
    Vector3d tv_rel = v_rel - v_rel.dot(contact_norm) * contact_norm;
    double tv_rel_n = tv_rel.norm();

    double gamma;
    Vector3d tv_rel_u;
    if (tv_rel_n == 0) {
        friction_forces.setZero();
        fric_jaco_type = 0;
        return;
    } else if (tv_rel_n > nu) {
        gamma = 1.0;
        fric_jaco_type = 1;
    } else {
        gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
        fric_jaco_type = 2;
    }
    tv_rel_u = tv_rel / tv_rel_n;

    Vector3d ffr_val = mu * gamma * tv_rel_u;

    friction_forces(seq(0, 2)) = ffr_val * f1s_n;
    friction_forces(seq(3, 5)) = ffr_val * f1e_n;
    friction_forces(seq(6, 8)) = -ffr_val * f2s_n;
    friction_forces(seq(9, 11)) = -ffr_val * f2e_n;
}


void contactPotentialIMC::computeFc(bool waitTime) {
    int edge1, edge2, edge3, edge4, constraintType;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->contact_ids(i, 0);
        edge2 = col_detector->contact_ids(i, 1);
        constraintType = col_detector->contact_ids(i, 2);
        edge3 = col_detector->contact_ids(i, 3);
        edge4 = col_detector->contact_ids(i, 4);

        prepContactInput(edge1, edge2, edge3, edge4, constraintType);
        contact_gradient.setZero();

        if (constraintType == 0)
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_p2p_gradient_func.call(p2p_gradient.data(), p2p_input.data());
            }
            else {
                sym_eqs->E_p2p_pen_gradient_func.call(p2p_gradient.data(), p2p_input.data());
            }

            // insert gradient and hessian to contact gradient and contact hessian
            contact_gradient(seq(0, 2)) = p2p_gradient(seq(0, 2));
            contact_gradient(seq(6, 8)) = p2p_gradient(seq(3, 5));
        }

        if (constraintType == 1)
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_e2p_gradient_func.call(e2p_gradient.data(), e2p_input.data());
            }
            else {
                sym_eqs->E_e2p_pen_gradient_func.call(e2p_gradient.data(), e2p_input.data());
            }

            // insert gradient and hessian to contact gradient and contact hessian
            contact_gradient(seq(0, 2)) = e2p_gradient(seq(0, 2));
            contact_gradient(seq(3, 5)) = e2p_gradient(seq(3, 5));
            contact_gradient(seq(6, 8)) = e2p_gradient(seq(6, 8));
        }

        if (constraintType == 2)
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_e2e_gradient_func.call(e2e_gradient.data(), e2e_input.data());
            }
            else {
                sym_eqs->E_e2e_pen_gradient_func.call(e2e_gradient.data(), e2e_input.data());
            }
            contact_gradient = e2e_gradient;
        }
        contact_gradient *= scale * contact_stiffness;

        // add friction
        if (friction && !waitTime) {
            prepFrictionInput(edge1, edge2, edge3, edge4);
            computeFriction(edge1, edge2, edge3, edge4);

            contact_gradient += friction_forces;
        }

        for (int e1 = 0; e1 < 3; e1++) {
            stepper->addForce(4 * edge1 + e1, contact_gradient[e1]);
            stepper->addForce(4 * edge3 + e1, contact_gradient[e1 + 3]);
            stepper->addForce(4 * edge2 + e1, contact_gradient[e1 + 6]);
            stepper->addForce(4 * edge4 + e1, contact_gradient[e1 + 9]);
        }
    }
}


void contactPotentialIMC::computeFcJc(bool waitTime) {
    int edge1, edge2, edge3, edge4, constraintType;
    for (int i = 0; i < col_detector->num_collisions; i++) {
        edge1 = col_detector->contact_ids(i, 0);
        edge2 = col_detector->contact_ids(i, 1);
        constraintType = col_detector->contact_ids(i, 2);
        edge3 = col_detector->contact_ids(i, 3);
        edge4 = col_detector->contact_ids(i, 4);

        prepContactInput(edge1, edge2, edge3, edge4, constraintType);
        contact_gradient.setZero();
        contact_hessian.setZero();

        if (constraintType == 0)
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_p2p_gradient_func.call(p2p_gradient.data(), p2p_input.data());
                sym_eqs->E_p2p_hessian_func.call(p2p_hessian.data(), p2p_input.data());
            }
            else {
                sym_eqs->E_p2p_pen_gradient_func.call(p2p_gradient.data(), p2p_input.data());
                sym_eqs->E_p2p_pen_hessian_func.call(p2p_hessian.data(), p2p_input.data());
            }

            // insert gradient and hessian to contact gradient and contact hessian
            contact_gradient(seq(0, 2)) = p2p_gradient(seq(0, 2));
            contact_gradient(seq(6, 8)) = p2p_gradient(seq(3, 5));
            contact_hessian.block<3, 3>(0, 0) = p2p_hessian.block<3, 3>(0, 0);
            contact_hessian.block<3, 3>(0, 6) = p2p_hessian.block<3, 3>(0, 3);
            contact_hessian.block<3, 3>(6, 0) = p2p_hessian.block<3, 3>(3, 0);
            contact_hessian.block<3, 3>(6, 6) = p2p_hessian.block<3, 3>(3, 3);
        }
        if (constraintType == 1)
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_e2p_gradient_func.call(e2p_gradient.data(), e2p_input.data());
                sym_eqs->E_e2p_hessian_func.call(e2p_hessian.data(), e2p_input.data());
            }
            else {
                sym_eqs->E_e2p_pen_gradient_func.call(e2p_gradient.data(), e2p_input.data());
                sym_eqs->E_e2p_pen_hessian_func.call(e2p_hessian.data(), e2p_input.data());
            }

            // insert gradient and hessian to contact gradient and contact hessian
            contact_gradient(seq(0, 2)) = e2p_gradient(seq(0, 2));
            contact_gradient(seq(3, 5)) = e2p_gradient(seq(3, 5));
            contact_gradient(seq(6, 8)) = e2p_gradient(seq(6, 8));
            contact_hessian.block<9, 9>(0, 0) = e2p_hessian;
        }
        if (constraintType == 2) // 0
        {
            if (col_detector->contact_ids(i, 5) == 0) {
                sym_eqs->E_e2e_gradient_func.call(e2e_gradient.data(), e2e_input.data());
                sym_eqs->E_e2e_hessian_func.call(e2e_hessian.data(), e2e_input.data());
            }
            else {
                sym_eqs->E_e2e_pen_gradient_func.call(e2e_gradient.data(), e2e_input.data());
                sym_eqs->E_e2e_pen_hessian_func.call(e2e_hessian.data(), e2e_input.data());
            }
            contact_gradient = e2e_gradient;
            contact_hessian = e2e_hessian;
        }

        contact_gradient *= scale * contact_stiffness;
        contact_hessian *= pow(scale, 2) * contact_stiffness;

        // add friction
        if (friction && !waitTime) {
            prepFrictionInput(edge1, edge2, edge3, edge4);
            computeFriction(edge1, edge2, edge3, edge4);

            if (fric_jaco_type == 1) {
                sym_eqs->friction_partials_gamma1_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
                sym_eqs->friction_partials_gamma1_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());
            }
            else if (fric_jaco_type == 2) {
                sym_eqs->friction_partials_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
                sym_eqs->friction_partials_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());
            }

            Matrix<double, 3, 12> zeroMatrix;
            zeroMatrix.setZero();

            if (constraintType == 0) {
                friction_partials_dfr_dfc.block<3, 12>(3, 0) = zeroMatrix;
                friction_partials_dfr_dfc.block<3, 12>(9, 0) = zeroMatrix;
            }

            if (constraintType == 1) {
                friction_partials_dfr_dfc.block<3, 12>(9, 0) = zeroMatrix;
            }

            if (fric_jaco_type == 0) {
                friction_jacobian.setZero();
            }
            else {
                friction_jacobian = friction_partials_dfr_dx + friction_partials_dfr_dfc.transpose() * contact_hessian;
            }

            contact_gradient += friction_forces;
            contact_hessian += friction_jacobian;
        }

        for (int e1 = 0; e1 < 3; e1++) {
            stepper->addForce(4 * edge1 + e1, contact_gradient[e1]);
            stepper->addForce(4 * edge3 + e1, contact_gradient[e1 + 3]);
            stepper->addForce(4 * edge2 + e1, contact_gradient[e1 + 6]);
            stepper->addForce(4 * edge4 + e1, contact_gradient[e1 + 9]);
        }

        // add hessian
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // first row
                stepper->addJacobian(4 * edge1 + i, 4 * edge1 + j, contact_hessian(j, i));
                stepper->addJacobian(4 * edge1 + i, 4 * edge3 + j, contact_hessian(3 + j, i));
                stepper->addJacobian(4 * edge1 + i, 4 * edge2 + j, contact_hessian(6 + j, i));
                stepper->addJacobian(4 * edge1 + i, 4 * edge4 + j, contact_hessian(9 + j, i));

                // second row
                stepper->addJacobian(4 * edge3 + i, 4 * edge1 + j, contact_hessian(j, 3 + i));
                stepper->addJacobian(4 * edge3 + i, 4 * edge3 + j, contact_hessian(3 + j, 3 + i));
                stepper->addJacobian(4 * edge3 + i, 4 * edge2 + j, contact_hessian(6 + j, 3 + i));
                stepper->addJacobian(4 * edge3 + i, 4 * edge4 + j, contact_hessian(9 + j, 3 + i));

                // third row
                stepper->addJacobian(4 * edge2 + i, 4 * edge1 + j, contact_hessian(j, 6 + i));
                stepper->addJacobian(4 * edge2 + i, 4 * edge3 + j, contact_hessian(3 + j, 6 + i));
                stepper->addJacobian(4 * edge2 + i, 4 * edge2 + j, contact_hessian(6 + j, 6 + i));
                stepper->addJacobian(4 * edge2 + i, 4 * edge4 + j, contact_hessian(9 + j, 6 + i));

                // forth row
                stepper->addJacobian(4 * edge4 + i, 4 * edge1 + j, contact_hessian(j, 9 + i));
                stepper->addJacobian(4 * edge4 + i, 4 * edge3 + j, contact_hessian(3 + j, 9 + i));
                stepper->addJacobian(4 * edge4 + i, 4 * edge2 + j, contact_hessian(6 + j, 9 + i));
                stepper->addJacobian(4 * edge4 + i, 4 * edge4 + j, contact_hessian(9 + j, 9 + i));
            }
        }
    }
}