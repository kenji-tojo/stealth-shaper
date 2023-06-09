#pragma once

#include "internal/math.hpp"


namespace stealth {

struct Hit {
public:
    float dist = std::numeric_limits<float>::max();
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f nrm = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f wo = Eigen::Vector3f::UnitZ();
    int obj_id = -1;
    int mat_id = -1;
    int prim_id = 0;

    operator bool() const { return obj_id >= 0; }
};

} // namespace stealth