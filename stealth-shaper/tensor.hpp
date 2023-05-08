#pragma once

#include <vector>
#include <memory>

#include "internal/math.hpp"


namespace stealth {

template<typename Matrix_ = Eigen::MatrixXf>
class Tensor {
public:
    using Matrix = Matrix_;
    using Scalar = typename Matrix::Scalar;

    std::shared_ptr<Matrix> mat;
    Matrix grad;

    [[nodiscard]] unsigned int rows() const { return mat->rows(); }
    [[nodiscard]] unsigned int cols() const { return mat->cols(); }

    explicit Tensor(const std::shared_ptr<Matrix> &_mat) {
        assert(_mat);
        mat = _mat;
        grad.setZero(rows(), cols());
    }

    virtual void descent(Scalar alpha) {
        *mat -= alpha * grad;
        grad.setZero();
    }

protected:
    Tensor() = default;

};


template<typename MatrixV, typename MatrixF, typename MatrixN, typename Vector3>
struct GradientAccumulatorVertex{
public:
    using TensorV = Tensor<MatrixV>;

    int obj_id = -1;

    std::shared_ptr<TensorV> V;
    std::shared_ptr<MatrixV> grad = std::make_shared<MatrixV>();
    std::shared_ptr<MatrixF> F;
    std::shared_ptr<MatrixN> triN;

    GradientAccumulatorVertex(
            const std::shared_ptr<TensorV> &V,
            const std::shared_ptr<MatrixF> &F,
            const std::shared_ptr<MatrixN> &triN,
            int obj_id) {

        this->obj_id = obj_id;
        assert(this->obj_id >= 0);

        assert(V);
        assert(F);
        assert(triN);
        this->V = V;
        this->F = F;
        this->triN = triN;

        int nV = this->V->rows();
        grad->setZero(nV, 3);
    }

    void accumulate(
            unsigned int tri_id,
            const typename Vector3::Scalar weight,
            Vector3 &&dN) {

        const unsigned int ip0 = (*F)(tri_id, 0);
        const unsigned int ip1 = (*F)(tri_id, 1);
        const unsigned int ip2 = (*F)(tri_id, 2);

        const Vector3 p0 = V->mat->row(ip0).transpose();
        const Vector3 p1 = V->mat->row(ip1).transpose();
        const Vector3 p2 = V->mat->row(ip2).transpose();

        const auto dblA = (p1-p0).cross(p2-p0).norm();

        const Vector3 nrm = triN->row(tri_id).transpose();

        const Vector3 h0 = (p2-p1).cross(nrm) / dblA;
        const Vector3 h1 = (p0-p2).cross(nrm) / dblA;
        const Vector3 h2 = (p1-p0).cross(nrm) / dblA;

        grad->row(ip0) += weight * h0.dot(dN) * nrm;
        grad->row(ip1) += weight * h1.dot(dN) * nrm;
        grad->row(ip2) += weight * h2.dot(dN) * nrm;
    }

    void flush() { V->grad += *grad; }

};

} // namespace stealth