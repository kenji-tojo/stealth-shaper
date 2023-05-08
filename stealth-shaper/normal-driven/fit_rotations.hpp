#pragma once

namespace stealth {
namespace {

inline void fit_rotations(const Eigen::MatrixXd &S, Eigen::MatrixXd &R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixV() * svd.matrixU().transpose();
    assert(!std::isnan(R.determinant()));
    assert(R.determinant() != 0);
    if (R.determinant() < 0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.rightCols(1) = U.rightCols(1) * -1;
        R = svd.matrixV() * U.transpose();
    }
    assert(R.determinant() > 0);
}

} // namespace
} // namespace stealth