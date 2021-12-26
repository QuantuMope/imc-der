#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"


class collisionDetector
{
public:
    collisionDetector(elasticRod &m_rod, timeStepper &m_stepper, double m_collision_limit);

    void detectCollisions();

    MatrixXi edge_ids;
    MatrixXi candidate_ids;
    int num_collisions;
    double min_dist;

private:
    elasticRod* rod;
    timeStepper* stepper;
    double collision_limit;
    int num_edge_combos;
    double scale;
    double contact_limit;

    void fixbound(double &x);
    void computeMinDistance(const Vector3d &v1s, const Vector3d &v1e, const Vector3d &v2s, const Vector3d &v2e, double& dist);
};