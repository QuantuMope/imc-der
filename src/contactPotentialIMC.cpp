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

    velo_limit = 1e-8;

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

    eye_mat << 1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0;

    sym_eqs = new symbolicEquations();
    sym_eqs->generateContactPotentialPiecewiseFunctions1();
    sym_eqs->generateContactPotentialPiecewiseFunctions2();
    if (friction) {
        sym_eqs->generateFrictionJacobianPiecewiseFunctions1();
        sym_eqs->generateFrictionJacobianPiecewiseFunctions2();
        sym_eqs->generateFrictionJacobianPiecewiseFunctions3();
        sym_eqs->generateFrictionJacobianPiecewiseFunctions4();
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

    friction_input2[3] = contact_gradient(0);
    friction_input2[4] = contact_gradient(1);
    friction_input2[5] = contact_gradient(2);
    friction_input2[6] = contact_gradient(3);
    friction_input2[7] = contact_gradient(4);
    friction_input2[8] = contact_gradient(5);
    friction_input2[9] = contact_gradient(6);
    friction_input2[10] = contact_gradient(7);
    friction_input2[11] = contact_gradient(8);
    friction_input2[12] = contact_gradient(9);
    friction_input2[13] = contact_gradient(10);
    friction_input2[14] = contact_gradient(11);
    friction_input2[16] = mu;
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
    tv_rel = v_rel - v_rel.dot(contact_norm) * contact_norm;
    tv_rel_n = tv_rel.norm();

    double gamma;
    if (tv_rel_n == 0) {
        gamma = 0;
        tv_rel_u.setZero();
        fric_jaco_type = 0;
    }
    else if (tv_rel_n < velo_limit) {
        gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
        tv_rel_u = tv_rel / tv_rel_n ;
        fric_jaco_type = 0;
    }
    else if (tv_rel_n > nu) {
        gamma = 1.0;
        tv_rel_u = tv_rel / tv_rel_n;
        fric_jaco_type = 1;
    } else {
        gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
        tv_rel_u = tv_rel / tv_rel_n;
        fric_jaco_type = 2;
    }
    friction_input2[0] = tv_rel_u[0];
    friction_input2[1] = tv_rel_u[1];
    friction_input2[2] = tv_rel_u[2];
    friction_input2[15] = gamma;
    dgamma_input[0] = tv_rel_n;
    dgamma_input[1] = K2;

    Vector3d ffr_val = mu * gamma * tv_rel_u;

    if (gamma == 0)
    {
      friction_input2[15] = 1e-5;
    }

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


MatrixXd contactPotentialIMC::computeFEM(const int edge1, const int edge2, const int edge3, const int edge4, int idx, int constraintType) {
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
    tv_rel = v_rel - v_rel.dot(contact_norm) * contact_norm;
    tv_rel_n = tv_rel.norm();

    double gamma;
    if (tv_rel_n == 0) {
        gamma = 0;
        tv_rel_u.setZero();
    }
    else if (tv_rel_n < velo_limit) {
        gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
        tv_rel_u = tv_rel / tv_rel_n ;
    }
    else if (tv_rel_n > nu) {
        gamma = 1.0;
        tv_rel_u = tv_rel / tv_rel_n;
    } else {
        gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
        tv_rel_u = tv_rel / tv_rel_n;
    }

    Vector3d ffr_val = mu * gamma * tv_rel_u;

    VectorXd friction1;
    friction1.setZero(12);


    friction1(seq(0, 2)) = ffr_val * f1s_n;
    friction1(seq(3, 5)) = ffr_val * f1e_n;
    friction1(seq(6, 8)) = -ffr_val * f2s_n;
    friction1(seq(9, 11)) = -ffr_val * f2e_n;

    double h = 1e-14;
    MatrixXd FEM;
    FEM = MatrixXd::Zero(12, 12);

    for (int i = 0; i < 12; i ++)
    {
      for (int j = 0; j< 12; j++)
      {
        x1s = rod->getVertex(edge1);
        x1e = rod->getVertex(edge3);
        x2s = rod->getVertex(edge2);
        x2e = rod->getVertex(edge4);
        // store the values diff(Fr(i)/diffh)

        int idx1 = j/3;
        int idx2 = j%3;

        switch (idx1){
          case 0:{
            x1s(idx2) = x1s(idx2)+h;
            break;
          }
          case 1:{
            x1e(idx2) = x1e(idx2)+h;
            break;
          }
          case 2:{
            x2s(idx2) = x2s(idx2)+h;
            break;
          }
          case 3:{
            x2e(idx2) = x2e(idx2)+h;
            break;
          }
        }

        // update contact gradient
        // if (constraintType == 0) //p2p
        // {
        //     p2p_input[0] = x1s(0) * scale;
        //     p2p_input[1] = x1s(1) * scale;
        //     p2p_input[2] = x1s(2) * scale;
        //
        //     p2p_input[3] = x2s(0) * scale;
        //     p2p_input[4] = x2s(1) * scale;
        //     p2p_input[5] = x2s(2) * scale;
        // }
        // else if (constraintType == 1) //e2p
        // {
        //     e2p_input[0] = x1s(0) * scale;
        //     e2p_input[1] = x1s(1) * scale;
        //     e2p_input[2] = x1s(2) * scale;
        //     e2p_input[3] = x1e(0) * scale;
        //     e2p_input[4] = x1e(1) * scale;
        //     e2p_input[5] = x1e(2) * scale;
        //     e2p_input[6] = x2s(0) * scale;
        //     e2p_input[7] = x2s(1) * scale;
        //     e2p_input[8] = x2s(2) * scale;
        // }
        // else if (constraintType == 2) //e2e
        // {
        //     e2e_input[0] = x1s(0) * scale;
        //     e2e_input[1] = x1s(1) * scale;
        //     e2e_input[2] = x1s(2) * scale;
        //     e2e_input[3] = x1e(0) * scale;
        //     e2e_input[4] = x1e(1) * scale;
        //     e2e_input[5] = x1e(2) * scale;
        //     e2e_input[6] = x2s(0) * scale;
        //     e2e_input[7] = x2s(1) * scale;
        //     e2e_input[8] = x2s(2) * scale;
        //     e2e_input[9] = x2e(0) * scale;
        //     e2e_input[10] = x2e(1) * scale;
        //     e2e_input[11] = x2e(2) * scale;
        // }
        // VectorXd contact_gradient1;
        // contact_gradient1.setZero(12);
        //
        //
        // if (constraintType == 0)
        // {
        //     if (col_detector->contact_ids(idx, 5) == 0) {
        //         sym_eqs->E_p2p_gradient_func.call(p2p_gradient.data(), p2p_input.data());
        //     }
        //     else {
        //         sym_eqs->E_p2p_pen_gradient_func.call(p2p_gradient.data(), p2p_input.data());
        //     }
        //
        //     // insert gradient and hessian to contact gradient and contact hessian
        //     contact_gradient1(seq(0, 2)) = p2p_gradient(seq(0, 2));
        //     contact_gradient1(seq(6, 8)) = p2p_gradient(seq(3, 5));
        // }
        // if (constraintType == 1)
        // {
        //     if (col_detector->contact_ids(idx, 5) == 0) {
        //         sym_eqs->E_e2p_gradient_func.call(e2p_gradient.data(), e2p_input.data());
        //     }
        //     else {
        //         sym_eqs->E_e2p_pen_gradient_func.call(e2p_gradient.data(), e2p_input.data());
        //     }
        //
        //     // insert gradient and hessian to contact gradient and contact hessian
        //     contact_gradient1(seq(0, 2)) = e2p_gradient(seq(0, 2));
        //     contact_gradient1(seq(3, 5)) = e2p_gradient(seq(3, 5));
        //     contact_gradient1(seq(6, 8)) = e2p_gradient(seq(6, 8));
        // }
        // if (constraintType == 2) // 0
        // {
        //     if (col_detector->contact_ids(idx, 5) == 0) {
        //         sym_eqs->E_e2e_gradient_func.call(e2e_gradient.data(), e2e_input.data());
        //     }
        //     else {
        //         sym_eqs->E_e2e_pen_gradient_func.call(e2e_gradient.data(), e2e_input.data());
        //     }
        //     contact_gradient1 = e2e_gradient;
        // }
        //
        // contact_gradient1 *= scale * contact_stiffness;
        //
        //
        // f1s = contact_gradient1(seq(0, 2));
        // f1e = contact_gradient1(seq(3, 5));
        // f2s = contact_gradient1(seq(6, 8));
        // f2e = contact_gradient1(seq(9, 11));
        //
        // f1s_n = f1s.norm();
        // f1e_n = f1e.norm();
        // f2s_n = f2s.norm();
        // f2e_n = f2e.norm();
        //
        // fn = (f1s + f1e).norm();
        //
        // beta11 = f1s_n / fn;
        // beta21 = f2s_n / fn;
        //
        // if (beta11 > 1) beta11 = 1;
        // if (beta11 < 0) beta11 = 0;
        // if (beta21 > 1) beta21 = 1;
        // if (beta21 < 0) beta21 = 0;
        //
        // beta12 = 1 - beta11;
        // beta22 = 1 - beta21;

        // compute correct forces
        v1s = (x1s - x1s0) / rod->dt;
        v1e = (x1e - x1e0) / rod->dt;
        v2s = (x2s - x2s0) / rod->dt;
        v2e = (x2e - x2e0) / rod->dt;

        v1 = beta11 * v1s + beta12 * v1e;
        v2 = beta21 * v2s + beta22 * v2e;
        v_rel = v1 - v2;

        tv_rel = v_rel - v_rel.dot(contact_norm) * contact_norm;
        tv_rel_n = tv_rel.norm();

        if (tv_rel_n == 0) {
            gamma = 0;
            tv_rel_u.setZero();
        }
        else if (tv_rel_n < velo_limit) {
            gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
            tv_rel_u = tv_rel / tv_rel_n ;
        }
        else if (tv_rel_n > nu) {
            gamma = 1.0;
            tv_rel_u = tv_rel / tv_rel_n;
        } else {
            gamma = (2.0 / (1 + exp(-K2 * tv_rel_n))) - 1;
            tv_rel_u = tv_rel / tv_rel_n;
        }
        ffr_val = mu * gamma * tv_rel_u;

        VectorXd friction2;
        friction2.setZero(12);


        friction2(seq(0, 2)) = ffr_val * f1s_n;
        friction2(seq(3, 5)) = ffr_val * f1e_n;
        friction2(seq(6, 8)) = -ffr_val * f2s_n;
        friction2(seq(9, 11)) = -ffr_val * f2e_n;

        FEM(i, j) = (friction2(i) - friction1(i))/h;
      }
    }

    // cout <<"FEM: " << endl;
    // cout << FEM << endl;

    return FEM;


}


void contactPotentialIMC::computeFcJc(bool waitTime) {
    int edge1, edge2, edge3, edge4, constraintType;

    int zeroCount = 0;
    int oneCount = 0;
    int twoCount = 0;


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

            sym_eqs->dtv_rel_dfc_func.call(dtv_rel_dfc_T.data(), friction_input.data());
            sym_eqs->dtv_rel_dx_func.call(dtv_rel_dx_T.data(), friction_input.data());
            sym_eqs->dgamma_dtv_rel_n_func.call(dgamma_dtv_rel_n.data(), dgamma_input.data());

            sym_eqs->dfr_dgamma_func.call(dfr_dgamma.data(), friction_input2.data());
            sym_eqs->dfr_dtv_rel_u_func.call(dfr_dtv_rel_u_T.data(), friction_input2.data());
            sym_eqs->dfr_dfc_func.call(dfr_dfc.data(), friction_input2.data());

            // SymEngine for some reason outputs these matrices in transposed order.
            // Doesn't affect symmetric matrices, but we have to be careful about non-symmetric.
            dtv_rel_dfc = dtv_rel_dfc_T.transpose();
            dtv_rel_dx = dtv_rel_dx_T.transpose();
            dfr_dtv_rel_u = dfr_dtv_rel_u_T.transpose();

//            if (fric_jaco_type == 1) {
//                sym_eqs->friction_partials_gamma1_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
//                sym_eqs->friction_partials_gamma1_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());
//            }
//            else if (fric_jaco_type == 2) {
//                sym_eqs->dtv_rel_dfc_func.call(dtv_rel_dfc.data(), friction_input.data());
//                sym_eqs->dtv_rel_dx_func.call(dtv_rel_dx.data(), friction_input.data());
//                sym_eqs->dgamma_dtv_rel_n_func.call(dgamma_dtv_rel_n.data(), dgamma_input.data());
//
//                sym_eqs->dfr_dgamma_func.call(dfr_dgamma.data(), friction_input2.data());
//                sym_eqs->dfr_dtv_rel_u_func.call(dfr_dtv_rel_u.data(), friction_input2.data());
//                sym_eqs->dfr_dfc_func.call(dfr_dfc.data(), friction_input2.data());
//            }

            Matrix<double, 3, 12> zeroMatrix1;
            Matrix<double, 3, 3> zeroMatrix2;
            zeroMatrix1.setZero();
            zeroMatrix2.setZero();

//            cout << dfr_dfc << endl;

            if (constraintType == 0) {
                dfr_dfc.block<3, 12>(3, 0) = zeroMatrix1;
                dfr_dfc.block<3, 12>(9, 0) = zeroMatrix1;
                dtv_rel_dfc.block<3, 3>(0, 3) = zeroMatrix2;
                dtv_rel_dfc.block<3, 3>(0, 9) = zeroMatrix2;
            }

            if (constraintType == 1) {
                dfr_dfc.block<3, 12>(9, 0) = zeroMatrix1;
                dtv_rel_dfc.block<3, 3>(0, 9) = zeroMatrix2;
            }

//            cout << "dtv_rel_dfc" << endl;
//            cout << dtv_rel_dfc << endl;
//            cout << "dtv_rel_dx" << endl;
//            cout << dtv_rel_dx << endl;
//            cout << "dgamma_dtv_rel_n" << endl;
//            cout << dgamma_dtv_rel_n << endl;
//            cout << "dfr_dgamma" << endl;
//            cout << dfr_dgamma << endl;
//            cout << "dfr_dtv_rel_u" << endl;
//            cout << dfr_dtv_rel_u << endl;
//            cout << "dfr_dfc" << endl;
//            cout << dfr_dfc << endl;
//            exit(0);

            if (fric_jaco_type == 0) {
                dtv_rel_dx_cr = dtv_rel_dfc * contact_hessian + dtv_rel_dx;

                dtv_rel_u_dx_cr = (eye_mat - tv_rel_u * tv_rel_u.transpose()) * (1.0 / velo_limit) * dtv_rel_dx_cr;

                dgamma_dx_cr = dgamma_dtv_rel_n * (tv_rel.transpose() / velo_limit) * dtv_rel_dx_cr;

                friction_jacobian = dfr_dtv_rel_u * dtv_rel_u_dx_cr + dfr_dgamma * dgamma_dx_cr + dfr_dfc.transpose() * contact_hessian;
                // cout << dfr_dtv_rel_u << endl;
                // cout << dtv_rel_u_dx_cr << endl;
                // cout << dfr_dgamma<<endl;
                // cout << dgamma_dx_cr<<endl;
                // cout << dfr_dfc<<endl;
                // exit(0);


                zeroCount = zeroCount + 1;

            }
            else if (fric_jaco_type == 1) {
                dtv_rel_dx_cr = dtv_rel_dfc * contact_hessian + dtv_rel_dx;

                dtv_rel_u_dx_cr = (eye_mat - tv_rel_u * tv_rel_u.transpose()) * (1 / tv_rel_n) * dtv_rel_dx_cr;

                friction_jacobian = dfr_dtv_rel_u * dtv_rel_u_dx_cr + dfr_dfc.transpose() * contact_hessian;
                oneCount = oneCount + 1;

            }
            else if (fric_jaco_type == 2) {
                dtv_rel_dx_cr = dtv_rel_dfc * contact_hessian + dtv_rel_dx;

                dtv_rel_u_dx_cr = (eye_mat - tv_rel_u * tv_rel_u.transpose()) * (1 / tv_rel_n) * dtv_rel_dx_cr;

                dgamma_dx_cr = dgamma_dtv_rel_n * tv_rel_u.transpose()  * dtv_rel_dx_cr;

                friction_jacobian = dfr_dtv_rel_u * dtv_rel_u_dx_cr + dfr_dgamma * dgamma_dx_cr + dfr_dfc.transpose() * contact_hessian;
                twoCount = twoCount + 1;

            }
            if (friction_jacobian.sum() != friction_jacobian.sum()) {
                cout << "NAN DETECTED" << endl;
            }
            // cout <<"method 1 "<<endl;
            // cout << friction_jacobian << endl;

            // previous method
            // friction_jacobian.setZero();
            // if (fric_jaco_type == 1) {
            //     sym_eqs->friction_partials_gamma1_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
            //     sym_eqs->friction_partials_gamma1_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());
            // }
            // else if (fric_jaco_type == 2) {
            //     sym_eqs->friction_partials_dfr_dx_func.call(friction_partials_dfr_dx.data(), friction_input.data());
            //     sym_eqs->friction_partials_dfr_dfc_func.call(friction_partials_dfr_dfc.data(), friction_input.data());
            // }
            // //
            // Matrix<double, 3, 12> zeroMatrix;
            // zeroMatrix.setZero();
            //
            // if (constraintType == 0) {
            //     friction_partials_dfr_dfc.block<3, 12>(3, 0) = zeroMatrix;
            //     friction_partials_dfr_dfc.block<3, 12>(9, 0) = zeroMatrix;
            // }
            //
            // if (constraintType == 1) {
            //     friction_partials_dfr_dfc.block<3, 12>(9, 0) = zeroMatrix;
            // }
            //
            // if (fric_jaco_type == 0) {
            //     friction_jacobian.setZero();
            // }
            // else {
            //     friction_jacobian = friction_partials_dfr_dx + friction_partials_dfr_dfc.transpose() * contact_hessian;
            // }
            // //
            // // cout <<"method 2 "<<endl;
            // // cout << friction_jacobian << endl;
            //
            // if (fric_jaco_type == 0)
            // {
            //   friction_jacobian = computeFEM(edge1, edge2, edge3, edge4, i, constraintType);
            // }
            // friction_jacobian = computeFEM(edge1, edge2, edge3, edge4, i, constraintType);

            // FEM method
            // MatrixXd friction_jacobian1 = computeFEM(edge1, edge2, edge3, edge4, i, constraintType);

            // check the difference between FEM and friction_jacobian
            // cout << (friction_jacobian - friction_jacobian1).norm()<<endl;

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

    // if (friction && !waitTime) {
    //   cout << "zeroCount "<<zeroCount << endl;
    //   cout << "oneCount "<<oneCount << endl;
    //   cout << "twoCount "<<twoCount << endl;
    //
    // }

}
