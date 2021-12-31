#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"


enum ParallelCase {
    NOPA,
    ACBD,
    ADBC,
    CADB,
    DACB
};


class collisionDetector
{
public:
    collisionDetector(elasticRod &m_rod, timeStepper &m_stepper, double m_collision_limit);

    void detectCollisions();
    void detectParallelCases();

    MatrixXi edge_ids;
    MatrixXi candidate_ids;
    vector<ParallelCase> parallel_cases;
    int num_collisions;
    double min_dist;

private:
    elasticRod* rod;
    timeStepper* stepper;
    double collision_limit;
    int num_edge_combos;
    double scale;
    double contact_limit;
    Vector3d axis_ref;  // used for 2d projection of edges
    Matrix3d eye;

    void fixbound(double &x);
    void computeMinDistance(const Vector3d &v1s, const Vector3d &v1e, const Vector3d &v2s, const Vector3d &v2e, double& dist);
};