#pragma once

#include "internal/math.hpp"


namespace stealth {

struct Ray{
public:
    Ray(const Eigen::Vector3f &o, const Eigen::Vector3f &d) {
        using namespace Eigen;
        org = Vector3f{o[0], o[1], o[2]};
        dir = Vector3f{d[0], d[1], d[2]};
    }

    Ray() = default;

    Eigen::Vector3f org;
    Eigen::Vector3f dir;
    const float tmin = 1e-4f;
    const float tmax = std::numeric_limits<float>::max();
};

} // namespace stealth