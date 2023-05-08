#pragma once

#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/unique_edge_map.h>
#include <igl/flip_edge.h>

#include "normal_driven_data.h"
#include "normal_driven_precomputation.h"
#include "normal_driven_prim_single_iteration.h"

#include <ctime>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_set>


namespace stealth {


template<typename DerivedV, typename DerivedF, typename DerivedN>
void vertex_target_normals(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixBase<DerivedF> &F,
        const Eigen::MatrixBase<DerivedN> &tri_tarN,
        Eigen::PlainObjectBase<DerivedN> &vrt_tarN) {
    using namespace Eigen;
    using Scalar = typename DerivedV::Scalar;

    int nV = V.rows();
    int nF = F.rows();

    vrt_tarN.setZero(nV,3);

    Vector<Scalar, Dynamic> A;
    igl::doublearea(V,F,A);

    A *= 2000.0 / A.sum(); // for improving numerics

    for (int kk = 0; kk < nF; ++kk) {
        int ip0 = F(kk,0);
        int ip1 = F(kk,1);
        int ip2 = F(kk,2);
        vrt_tarN.row(ip0) += A[kk] * tri_tarN.row(kk);
        vrt_tarN.row(ip1) += A[kk] * tri_tarN.row(kk);
        vrt_tarN.row(ip2) += A[kk] * tri_tarN.row(kk);
    }

    for (int ii = 0; ii < nV; ++ii) {
#if !defined(NDEBUG)
        const auto norm = vrt_tarN.row(ii).norm();
        assert(norm > 1e-5);
#endif
        vrt_tarN.row(ii).normalize();
    }
}


struct NormalDriven {
public:
    int type = 0; // 0: vertex, 1: face
    double lambda = 1.;
    int maxIter = 20;

    normal_driven_data data;

    std::unordered_set<int> IP;

    template<typename IndexList>
    void precomp(
            const Eigen::MatrixXd &refV,
            const Eigen::MatrixXi &F,
            IndexList &&_pinned) {

        using namespace std;
        using namespace Eigen;

        if constexpr (std::is_same_v<IndexList, std::unordered_set<int>>) {
            if (IP.empty() || !_pinned.empty()) { IP = std::forward<IndexList>(_pinned); }
        }
        else {
            // assuming that IndexList is ndarray with ndim == 1
            IP.clear();
            for (int ii = 0; ii < _pinned.shape(0); ++ii) { IP.insert(_pinned(ii)); }
        }

        this->nV_ = refV.rows();
        this->nF_ = F.rows();

        // prepare data for stylization
        data.lambda = lambda; // weighting for normal term

        // pin down at least one vertex
        {
            if (IP.empty()) {
                cout << "NormalDriven::precomp: no pinned vertex is specified, picking one randomly" << endl;
                IP.insert(F(0,0));
            }

            cout << "NormalDriven::precomp: pinning down " << IP.size() << " vertices" << endl;
            data.bc.resize(IP.size(), 3);
            data.b.resize(IP.size());
            int ii = 0;
            for (const auto ip: IP) {
                assert(ip >= 0);
                data.bc.row(ii) = refV.row(ip);
                data.b[ii] = ip;
                ii += 1;
            }
        }

        // precomputation
        normal_driven_precomputation(refV,F,type,data);
    }

    void precomp(const Eigen::MatrixXd &refV, const Eigen::MatrixXi &F) { precomp(refV, F, std::unordered_set<int>{}); }

    void update_vertices(
            const Eigen::MatrixXd &refV,
            const Eigen::MatrixXi &F,
            const Eigen::MatrixXd &tarN,
            Eigen::MatrixXd &defV) {

        using namespace std;
        using namespace Eigen;

        if (refV.rows() != nV_ || F.rows() != nF_) {
            cerr << "NormalDriven::update_vertices [ Error ]: precomp is needed" << endl;
            assert(false);
            return;
        }

        if (tarN.rows() != nF_) {
            cerr << "NormalDriven::update_vertices [ Error ]: invalid target normals" << endl;
            assert(false);
            return;
        }

        if (defV.rows()!=refV.rows() || defV.cols()!=3) {
            assert(false);
            cout << "NormalDriven::update_vertices [ Warning ]: incorrect size of defV, initializing it with refV" << endl;
            defV = refV;
        }

        data.lambda = lambda;

        MatrixXd vrt_tarN;
        if (type == 0) {
            vertex_target_normals(refV, F, tarN, vrt_tarN);
        }

        // primtitive stylization
        for (int iter=0; iter<maxIter; iter++) {

            if (type == 0) {
                normal_driven_prim_single_iteration(refV,F,vrt_tarN,defV,data);
            }
            else if (type == 1) {
                normal_driven_prim_single_iteration(refV,F,tarN,defV,data);
            }
            else {
                assert(false);
            }

//            cout << "iteration: " << iter << ", reldV: " << data.reldV << endl;
//            if (data.reldV < stopReldV) break;
        }
        cout << "iteration: " << maxIter-1 << ", reldV: " << data.reldV << endl;
    }

    [[nodiscard]] int nV() const { return nV_; }
    [[nodiscard]] int nF() const { return nF_; }

private:
    int nV_ = -1;
    int nF_ = -1;

};


template<typename Matrix, typename Vector3>
void snap_normals(
        const std::vector<Vector3> &desired_normals,
        const Matrix &normals,
        Matrix &out) {
    assert(!desired_normals.empty());
    assert(normals.rows()>0 && normals.cols()==3);

    out.resize(normals.rows(), 3);

    for (int kk = 0; kk < normals.rows(); ++kk) {
        double max_proj = -1.;
        int argmax_id = 0;
        for (int ii = 0; ii < desired_normals.size(); ++ii) {
            const auto &nrm = desired_normals[ii];
            const double proj = normals.row(kk) * nrm;
            if (proj > max_proj) {
                max_proj = proj;
                argmax_id = ii;
            }
        }
        assert(max_proj > 0.);
        out.row(kk) = desired_normals[argmax_id].transpose();
    }
}


} // namespace stealth