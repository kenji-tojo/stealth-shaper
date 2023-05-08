#pragma once

#include "internal/math.hpp"

#include "igl/doublearea.h"


namespace stealth::geometry {

template<typename DerivedV, typename DerivedF>
void barycentric_cell_areas(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixBase<DerivedF> &F,
        typename DerivedV::Scalar *A /* count == V.rows() */) {
    using namespace Eigen;
    using Scalar = typename DerivedV::Scalar;

    const size_t n = V.rows();
    const size_t m = F.rows();

    Vector<Scalar, Dynamic> dblA;
    igl::doublearea(V, F, dblA);

    std::fill(A, A+n, Scalar(0.));

    for (int kk = 0; kk < m; ++kk) {
        A[F(kk,0)] += dblA[kk];
        A[F(kk,1)] += dblA[kk];
        A[F(kk,2)] += dblA[kk];
    }

    for (int ii = 0; ii < n; ++ii) {
        A[ii] /= Scalar(6.0);
    }
}


template<typename DerivedV>
double diameter(const Eigen::MatrixBase<DerivedV> &V) {
    const auto &minV = V.colwise().minCoeff();
    const auto &maxV = V.colwise().maxCoeff();
    return (maxV-minV).norm();
}


} // namespace stealth::geometry