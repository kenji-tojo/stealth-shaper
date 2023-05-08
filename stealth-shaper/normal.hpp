#pragma once

#include <cstdlib>
#include <iostream>

#include "tensor.hpp"


namespace stealth {


/*
 * tangent plane parameterization of the unit hemisphere
 * used for gradient-based optimization of surface normal vectors
 */
template<typename Matrix, typename Vector3>
class TensorUnit3XY: public Tensor<Matrix> {
public:
    using Scalar = typename Matrix::Scalar;

    Matrix frames_up;
    Matrix frames_b1;
    Matrix frames_b2;
    Matrix coords;

    explicit TensorUnit3XY(const std::shared_ptr<Matrix> &_mat) {
        assert(_mat);
        assert(_mat->cols() == 3);
        this->mat = _mat;
        this->grad.setZero(this->rows(), 2);
        reset_frames();
    }

    void descent(Scalar alpha) override {
        auto &g = this->grad;
        for (int ii = 0; ii < this->rows(); ++ii) {
            g(ii,0) = std::isnan(g(ii,0)) ? 0. : g(ii,0);
            g(ii,1) = std::isnan(g(ii,1)) ? 0. : g(ii,1);
        }
        coords -= alpha * g;
        g.setZero();
        update_unit3();
    }

    void update_unit3() {
        Vector3 nrm, b1, b2;
        for (int ii = 0; ii < this->rows(); ++ii) {
            get_frame(ii, nrm, b1, b2);
            const auto x = coords(ii, 0);
            const auto y = coords(ii, 1);
            this->mat->row(ii) = x*b1 + y*b2 + nrm;
#if !defined(NDEBUG)
            const auto norm = this->mat->row(ii).norm();
            const auto L = std::sqrt(x*x + y*y + Scalar(1.));
            if (std::abs((norm-L)/norm) >= 1e-5) {
                std::cout << "TensorUnit3XY::update_unit3 [ Warning ]: vector norm mismatch, "
                          << "norm = " << norm << ", "
                          << "L = " << norm << ", "
                          << "|norm-L|/norm = " << std::abs((norm-L)/norm) << std::endl;
            }
            assert(!std::isnan(norm));
            assert(std::isinf(L) || std::abs((norm-L)/norm) < 1e-5);
#endif
            this->mat->row(ii).normalize();
        }
    }

    void reset_frames() {
        frames_up.resize(this->rows(), 3);
        frames_b1.resize(this->rows(), 3);
        frames_b2.resize(this->rows(), 3);
        coords.setZero(this->rows(), 2);

        Vector3 nrm, b1, b2;
        for (int ii = 0; ii < this->rows(); ++ii) {
            nrm = this->mat->row(ii).transpose();
            internal::math::create_local_frame(nrm, b1, b2);
            frames_up.row(ii) = nrm.transpose();
            frames_b1.row(ii) = b1.transpose();
            frames_b2.row(ii) = b2.transpose();
        }
    }

    void get_frame(int index, Vector3 &nrm, Vector3 &b1, Vector3 &b2) {
        nrm = frames_up.row(index).transpose();
        b1 = frames_b1.row(index).transpose();
        b2 = frames_b2.row(index).transpose();
    }


    /*
     * derivative of the tangent plane parameterization
     */
    void dUdXY(int index, Vector3 &dUdX, Vector3 &dUdY) {
        const auto x = coords(index, 0);
        const auto y = coords(index, 1);
        const auto L = std::sqrt(x*x + y*y + Scalar(1.));
        const auto L3 = L*L*L;

        Vector3 nrm, b1, b2;
        get_frame(index, nrm, b1, b2);

        dUdX = (y*y+1)*b1 -x*y*b2 -x*nrm / L3;
        dUdY = -x*y*b1 + (x*x+1)*b2 -y*nrm / L3;
    }

};


template<typename MatrixN, typename Vector3>
struct GradientAccumulatorNormal {
public:
    using TensorN = Tensor<MatrixN>;
    using TensorNXY = TensorUnit3XY<MatrixN, Vector3>;

    int obj_id = -1;

    std::shared_ptr<TensorNXY> normals;
    std::shared_ptr<MatrixN> grad = std::make_shared<MatrixN>();

    GradientAccumulatorNormal(const std::shared_ptr<TensorN> &parameters, int obj_id) {

        this->obj_id = obj_id;
        assert(this->obj_id >= 0);

        assert(parameters);
        auto nrm = std::dynamic_pointer_cast<TensorNXY>(parameters);
        if (!nrm) {
            std::cerr << "GradientAccumulatorNormal [ Error ]: wrong tensor type" << std::endl;
            assert(false);
            std::exit(EXIT_FAILURE);
        }
        normals = nrm;

        grad->setZero(normals->rows(), 2);
    }

    void accumulate(
            unsigned int index,
            const typename Vector3::Scalar weight,
            Vector3 &&dN) {

        Vector3 dNdX, dNdY;
        normals->dUdXY(index, dNdX, dNdY);

        auto &g = *grad;
        g(index, 0) += weight * dNdX.transpose() * dN;
        g(index, 1) += weight * dNdY.transpose() * dN;
    }

    void flush() { normals->grad += *grad; }

};

} // namespace stealth