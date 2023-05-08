#pragma once

#include <vector>

#include "internal/math.hpp"


namespace stealth {

template<typename DerivedV>
std::vector<int> vertices_below_level(
        const Eigen::MatrixBase<DerivedV> &V,
        double level) {
    double min_z = V.colwise().minCoeff()[2];
    std::vector<int> indices;
    for (int ii = 0; ii < V.rows(); ++ii) {
        if (V(ii,2) >= min_z + level) { continue; }
        indices.push_back(ii);
    }
    return indices;
}

} // namespace stealth