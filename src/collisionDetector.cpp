#include "collisionDetector.h"


collisionDetector::collisionDetector(elasticRod &m_rod, timeStepper &m_stepper, double m_collision_limit) {
    rod = &m_rod;
    stepper = &m_stepper;
    collision_limit = m_collision_limit;
    scale = 1 / rod->rodRadius;
    contact_limit = scale * (2 * rod->rodRadius + collision_limit);

    axis_ref << 1, 0, 0;
    eye << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;

    num_edge_combos = 0;
    int ignore_adjacent = 3;  // Here, we assume that no edge can collide with the next 3 adjacent edges on either side
    for (int i = 0; i < rod->ne; i++) {
        for (int j = i+ignore_adjacent+1; j < rod->ne; j++) {
            parallel_cases.push_back(NOPA);
            num_edge_combos++;
        }
    }

    candidate_ids.setZero(num_edge_combos, 2);
    edge_ids.resize(num_edge_combos, 2);

    int real_index = 0;
    for (int i = 0; i < rod->ne; i++) {
        for (int j = i+ignore_adjacent+1; j < rod->ne; j++) {
            edge_ids(real_index, 0) = i;
            edge_ids(real_index, 1) = j;
            real_index++;
        }
    }
}


void collisionDetector::fixbound(double &x) {
    if (x > 1) {
        x = 1;
    }
    else if (x < 0) {
        x = 0;
    }
}

void collisionDetector::computeMinDistance(const Vector3d &v1s, const Vector3d &v1e, const Vector3d &v2s,
                                           const Vector3d &v2e, double& dist) {
    Vector3d e1 = v1e - v1s;
    Vector3d e2 = v2e - v2s;
    Vector3d e12 = v2s - v1s;

    double D1 = e1.array().pow(2).sum();
    double D2 = e2.array().pow(2).sum();
    double R = (e1.array() * e2.array()).sum();
    double S1 = (e1.array() * e12.array()).sum();
    double S2 = (e2.array() * e12.array()).sum();

    double den = D1 * D2 - pow(R, 2);

    double t = 0.0;
    if (den != 0) {
        t = (S1 * D2 - S2 * R) / den;
    }
    fixbound(t);

    double u = (t * R - S2) / D2;

    double uf = u;
    fixbound(uf);

    if (uf != u) {
        t = (uf * R + S1) / D1;
    }
    fixbound(t);

    dist = pow((e1 * t - e2 * uf - e12).array().pow(2).sum(), 0.5);
}


void collisionDetector::detectParallelCases() {
    int edge1, edge2;
    Vector3d v1s, v1e, v2s, v2e, e1, e2, e1_u, e2_u, v;
    double angle, c, s, A, B, C, D;
    for (int i = 0; i < num_collisions; i++) {
        edge1 = candidate_ids(i, 0);
        edge2 = candidate_ids(i, 1);

        v1s = rod->getVertex(edge1);
        v1e = rod->getVertex(edge1+1);
        v2s = rod->getVertex(edge2);
        v2e = rod->getVertex(edge2+1);

        e1 = v1e - v1s;
        e2 = v2e - v2s;

        e1_u = e1.array() / e1.norm();
        e2_u = e2.array() / e2.norm();

        angle = abs(acos(e1_u.dot(e2_u)));
        angle = min(angle, M_PI - angle) * 180.0 / M_PI;

        if (angle > 1.0) {
            parallel_cases[i] = NOPA;
        } else {
            v = e1_u.cross(axis_ref);
            c = e1_u.dot(axis_ref);
            s = v.norm();

            Matrix3d kmat;
            kmat << 0, -v[2], v[1],
                    v[2], 0, -v[0],
                    -v[1], v[0], 0;

            Matrix3d mat = eye + kmat + kmat * kmat * ((1 - c) / v.array().pow(2).sum());

            A = 0;
            B = e1.norm();
            C = mat(0, all).dot((v2s - v1s));
            D = mat(0, all).dot((v2e - v1s));

            if (A < C && C < B && B < D) {
                parallel_cases[i] = ACBD;
            }
            else if (A < D && D < B && B < C) {
                parallel_cases[i] = ADBC;
            }
            else if (C < A && A < D && D < B) {
                parallel_cases[i] = CADB;
            }
            else if (D < A && A < C && C < B) {
                parallel_cases[i] = DACB;
            }
            else {
                parallel_cases[i] = NOPA;
            }
        }
    }
}



void collisionDetector::detectCollisions() {
    int edge1, edge2;
    double curr_dist;
    min_dist = 1e10;  // something arbitrarily large
    num_collisions = 0;

    for (int i = 0; i < num_edge_combos; i++) {
        edge1 = edge_ids(i, 0);
        edge2 = edge_ids(i, 1);
        computeMinDistance(rod->getVertex(edge1) * scale, rod->getVertex(edge1+1) * scale,
                           rod->getVertex(edge2) * scale, rod->getVertex(edge2+1) * scale, curr_dist);

        if (curr_dist < min_dist) {
            min_dist = curr_dist;
        }
        if (curr_dist < contact_limit) {
            candidate_ids(num_collisions, 0) = edge1;
            candidate_ids(num_collisions, 1) = edge2;
            num_collisions++;
        }
    }
    min_dist /= scale;
}
