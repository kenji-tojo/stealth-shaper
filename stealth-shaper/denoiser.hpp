#pragma once

#include <vector>
#include <iostream>

#include "internal/math.hpp"

#include <Eigen/Sparse>
#include <igl/edge_flaps.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/doublearea.h>


namespace stealth {


template<typename Scalar = double>
class L1Denoiser {
public:
    using SpMat = Eigen::SparseMatrix<Scalar>;
    using VectorX = Eigen::Vector<Scalar, Eigen::Dynamic>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    Scalar alpha = decltype(alpha)(500.);
    Scalar r = decltype(r)(5.);

    template<typename DerivedV>
    L1Denoiser(const Eigen::MatrixBase<DerivedV> &V, const Eigen::MatrixXi &F) {

        Eigen::MatrixXi EF, EI;
        Eigen::VectorXi EMAP;
        igl::edge_flaps(F, uE, EMAP, EF, EI);
        std::cout << "L1Denoiser: found " << uE.rows() << " unique edges" << std::endl;

        std::vector<Eigen::Triplet<Scalar, int>> nbl;
        for (int jj = 0; jj < uE.rows(); ++jj) {
            if (EF(jj,0)<0 || EF(jj,1)<0) {
                std::cout << "L1Denoiser [ Warning ]: found boundary edge" << std::endl;
                continue;
            }
            nbl.emplace_back(jj, EF(jj,0), 1.);
            nbl.emplace_back(jj, EF(jj,1), -1.);
        }
        nblMat.resize(uE.rows(), F.rows());
        nblMat.setFromTriplets(nbl.begin(), nbl.end());

        update_positions(V, F);
    }

    template<typename DerivedV>
    void update_positions(const Eigen::MatrixBase<DerivedV> &V, const Eigen::MatrixXi &F) {

        // scaling that doesn't affect the effect of other parameters
        // used only for improving numerics
        constexpr Scalar scaling = decltype(scaling)(100.0);

        eLenMat.resize(uE.rows(), uE.rows());
        eLenMat.setZero();
        double sumL = 0.;
        for (int jj = 0; jj < uE.rows(); ++jj) {
            const auto p0 = V.row(uE(jj,0));
            const auto p1 = V.row(uE(jj,1));
            const auto l = (p0-p1).norm();
            eLenMat.insert(jj,jj) = scaling * l;
        }

        Eigen::Vector<typename DerivedV::Scalar, Eigen::Dynamic> A;
        igl::doublearea(V, F, A);
        const auto totA = A.sum();
        areaMat.resize(F.rows(), F.rows());
        areaMat.setZero();
        for (int kk = 0; kk < F.rows(); ++kk) {
            areaMat.insert(kk,kk) = scaling * A[kk];
        }
    }

    template<typename DerivedN>
    void denoise(
            const Eigen::MatrixBase<DerivedN> &inN,
            Eigen::PlainObjectBase<DerivedN> &N,
            int n_iter) {

        update_solver();
        update_weight(inN);

        MatrixX lambda, P;

        lambda.setZero(uE.rows(), 3);
        P.setZero(uE.rows(), 3);
        N.setZero(inN.rows(), 3);

        for (int iter = 0; iter < n_iter; ++iter) {
            // N-sub problem
            solve_for_N(lambda, P, inN, N);

            for (int kk = 0; kk < N.rows(); kk++) {
                N.row(kk).normalize();
            }

            // P-sub problem
            solve_for_P(lambda, N, P);
            update_weight(N);

            // update lambda
            lambda = lambda + r * (P - nblMat * N);
        }

#ifndef NDEBUG
        for (int kk = 0; kk < N.rows(); ++kk) {
            double norm = N.row(kk).norm();
            assert(std::abs(norm-1.) < 1e-6);
        }
#endif
    }


    template<typename DerivedN, typename DerivedV>
    static void recover_positions(
            const Eigen::MatrixXi &F,
            const Eigen::MatrixBase<DerivedN> &tarN,
            Eigen::MatrixBase<DerivedV> &V /* the deformed positions will be updated in-place */) {

        std::cout << "L1Denoiser::recover_positions [ Warning ]: "
                  << "this is a debugging feature and shouldn't be used in the main code of Stealth Shaper"
                  << std::endl;

        std::vector<std::vector<int>> VF, VFi;
        igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

        for (int iter = 0; iter < 35; ++iter) {
            const DerivedV iniV = V;
            for (int ii = 0; ii < V.rows(); ii++) {
                const auto tau = Scalar(1.0) / Scalar(VF[ii].size());
                const Vector3 p = iniV.row(ii);
                for (int kk : VF[ii]) {
                    const Vector3 p0 = iniV.row(F(kk,0));
                    const Vector3 p1 = iniV.row(F(kk,1));
                    const Vector3 p2 = iniV.row(F(kk,2));
                    const Vector3 n = tarN.row(kk);
                    const Vector3 c = Scalar(1.0/3.0) * (p0+p1+p2);
                    V.row(ii) += tau * n.dot(c-p) * n;
                }
            }
        }
    }

private:

    Eigen::MatrixXi uE; // #E by 2 unique edge list stacking incident vertex indices

    SpMat nblMat; // #E by #F matrix discretizing the edge nabla operator
    SpMat eLenMat; // #E by #E matrix with edge lengths at its diagonal entries
    SpMat areaMat; // #F by #F matrix with face areas at its diagonal entries
    VectorX EW; // #E list of edge weights

    Eigen::SimplicialLDLT<SpMat> solver;
    MatrixX RHS; // an internal cache to avoid redundant memory allocation

    void update_solver() {
        assert(eLenMat.rows() > 0 && areaMat.rows() > 0);
        const auto systemMat = alpha * areaMat + r * nblMat.transpose() * eLenMat * nblMat;
        solver.analyzePattern(systemMat);
        solver.factorize(systemMat);
    }

    template<typename DerivedN>
    void update_weight(const Eigen::MatrixBase<DerivedN> &N) {
        EW.resize(uE.rows());
        const MatrixX nblN = nblMat * N;
        for (uint32_t jj = 0; jj < uE.rows(); ++jj) {
            const auto l2 = nblN.row(jj).squaredNorm();
            EW[jj] = std::exp(-1.0 * l2 * l2);
        }
    }

    template<typename DerivedL, typename DerivedN, typename DerivedI>
    void solve_for_P(
            const Eigen::MatrixBase<DerivedL> &lambda,
            const Eigen::MatrixBase<DerivedN> &N,
            Eigen::PlainObjectBase<DerivedI> &P) {
        assert(eLenMat.rows() > 0 && areaMat.rows() > 0);
        MatrixX W = nblMat * N;
        W -= (1.0 / r) * lambda;
        P.resize(uE.rows(), 3);
        P.setZero();
        for (uint32_t jj = 0; jj < uE.rows(); ++jj) {
            const Vector3 &w = W.row(jj);
            const auto wNorm = w.norm();
            if (wNorm > EW[jj] / r) {
                P.row(jj) = (1.0 - (EW[jj] / (r * wNorm))) * w;
            }
        }
    }

    template<typename DerivedL, typename DerivedI, typename DerivedN>
    void solve_for_N(
            const Eigen::MatrixBase<DerivedL> &lambda,
            const Eigen::MatrixBase<DerivedI>& P,
            const Eigen::MatrixBase<DerivedN> &inN,
            Eigen::PlainObjectBase<DerivedN> &N) {
        assert(eLenMat.rows() > 0 && areaMat.rows() > 0);
        RHS = alpha * areaMat * inN;
        RHS += nblMat.transpose() * eLenMat * (lambda + r * P);
        N = solver.solve(RHS);
    }

};


} // namespace stealth