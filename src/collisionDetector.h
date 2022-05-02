#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"


class collisionDetector
{
public:
    collisionDetector(elasticRod &m_rod, timeStepper &m_stepper, double m_delta, double m_col_limit);

    void constructCandidateSet();
    void detectCollisions();

    MatrixXi edge_ids;
    MatrixXi contact_ids;
    vector<Vector2i> candidateSet;
    int num_collisions;
    double min_dist;

private:
    elasticRod* rod;
    timeStepper* stepper;
    double delta;
    double col_limit;
    int num_edge_combos;
    double scale;
    double contact_limit;
    double candidate_limit;
    double numerical_limit;

    void fixbound(double &x);
    void computeMinDistance(const Vector3d &v1s, const Vector3d &v1e, const Vector3d &v2s, const Vector3d &v2e, double& dist);
    void computeMinDistance(int &idx1, int &idx2, int &idx3, int &idx4, double &dist, int &constraintType);
};