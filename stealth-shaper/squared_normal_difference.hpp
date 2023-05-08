#pragma once

#include <memory>

#include "internal/math.hpp"

#include "normal.hpp"
#include "trimesh.h"


namespace stealth {

template<typename TensorN>
void backprop_squared_normal_difference(
        const typename TensorN::Scalar beta,
        const TriMesh &mesh,
        const std::shared_ptr<TensorN> &normals) {

    TriMesh::Vector3 dN, dNdX, dNdY;
    for (int kk = 0; kk < normals->mat->rows(); ++kk) {
        normals->dUdXY(kk, dNdX, dNdY);
        dN = beta * mesh.A[kk] * (mesh.triN().row(kk) - mesh.refTriN.row(kk).cast<float>());
        normals->grad(kk, 0) += dNdX.transpose() * dN;
        normals->grad(kk, 1) += dNdY.transpose() * dN;
    }
}

} // namespace stealth