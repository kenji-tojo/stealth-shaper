#pragma once

#include <vector>
#include <unordered_set>

#include "internal/math.hpp"
#include "topology.hpp"

#include <igl/edge_flaps.h>
#include <igl/per_face_normals.h>


namespace stealth {


template<typename DerivedV, typename DerivedN, typename DerivedT, typename DerivedL>
void dihedral_angles(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixBase<DerivedN> &triN,
        const Eigen::MatrixXi &uE,
        const Eigen::MatrixXi &EF,
        Eigen::PlainObjectBase<DerivedT> &T /* #E list stacking | dihedral - \pi | values in radian */,
        Eigen::PlainObjectBase<DerivedL> &L /* #E list of | e | values */) {

    namespace math = internal::math;
    using Scalar = typename DerivedV::Scalar;
    using Vector3 = Eigen::Vector3<Scalar>;

    T.resize(uE.rows());
    L.resize(uE.rows());

    for (int jj = 0; jj < uE.rows(); ++jj) {
        const Vector3 &p0 = V.row(uE(jj,0));
        const Vector3 &p1 = V.row(uE(jj,1));
        const Vector3 &n0 = triN.row(EF(jj,0));
        const Vector3 &n1 = triN.row(EF(jj,1));
        const auto theta = std::acos(math::clip<Scalar>(n0.dot(n1), -1.0, 1.0));
        assert(theta >= 0. && theta <= M_PI);
        T[jj] = theta;
        L[jj] = (p0-p1).norm();
    }
}


template<typename DerivedV, typename DerivedG>
void geometric_criterion(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXi &uE,
        const Eigen::VectorXi &EMAP,
        const Eigen::MatrixXi &EF,
        Eigen::PlainObjectBase<DerivedG> &G /* #E list of geometric criterion values */) {

    using namespace Eigen;
    using Scalar = typename DerivedV::Scalar;

    Matrix<Scalar, Dynamic, 3> triN;
    igl::per_face_normals(V, F, triN);

    Vector<Scalar, Dynamic> T, L;
    dihedral_angles(V, triN, uE, EF, T, L);

    G.setZero(uE.rows());

    for (int iE = 0; iE < uE.rows(); ++iE) {
        int count = 0;
        for (int jj = 0; jj < 2; ++jj) {
            const int iF = EF(iE, jj);
            const int ip0 = F(iF,0);
            const int ip1 = F(iF,1);
            const int ip2 = F(iF,2);
            assert(ip0==uE(iE,0) || ip1==uE(iE,0) || ip2==uE(iE,0));
            assert(ip0==uE(iE,1) || ip1==uE(iE,1) || ip2==uE(iE,1));
            for (int kk = 0; kk < 3; ++kk) {
                const int jE = EMAP[kk*F.rows()+iF];
                assert(ip0==uE(jE,0) || ip1==uE(jE,0) || ip2==uE(jE,0));
                assert(ip0==uE(jE,1) || ip1==uE(jE,1) || ip2==uE(jE,1));
                if (jE == iE) continue;
                G[iE] += L[jE] * T[jE];
                count += 1;
            }
        }
        assert(count == 4);
    }
}


template<typename Scalar, typename DerivedR>
void reflectivity_criterion(
        const Eigen::MatrixXi &EF,
        const Scalar *triR /* #F list of per face reflectivity energy values */,
        Eigen::PlainObjectBase<DerivedR> &R /* #E list of reflectivity criterion values */) {

    R.setZero(EF.rows());

    for (int iE = 0; iE < EF.rows(); ++iE) {
        R[iE] += Scalar(.5) * triR[EF(iE,0)];
        R[iE] += Scalar(.5) * triR[EF(iE,1)];
    }
}


template<typename DerivedV, typename Scalar, typename DerivedC>
void split_criterion(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixXi &F,
        const Scalar *triR,
        Eigen::MatrixXi &uE,
        Eigen::MatrixXi &EF,
        Eigen::PlainObjectBase<DerivedC> &C) {

    using namespace Eigen;

    MatrixXi EI;
    VectorXi EMAP;
    igl::edge_flaps(F, uE, EMAP, EF, EI);

    Vector<Scalar, Dynamic> G, R;
    geometric_criterion(V, F, uE, EMAP, EF, G);
    reflectivity_criterion(EF, triR, R);

    C = G.cwiseProduct(R);
}


template<typename DerivedV, typename Scalar, typename DerivedC>
void subdivided_edges(
        const Eigen::MatrixBase<DerivedV> &V,
        const Eigen::MatrixXi &F,
        const Scalar *triR,
        const Scalar ratio,
        Eigen::MatrixXi &uE,
        Eigen::MatrixXi &EF,
        Eigen::PlainObjectBase<DerivedC> &C,
        std::vector<int> &SE /* list of edge indices to be subdivided */) {

    using namespace Eigen;
    namespace math = internal::math;

    split_criterion(V, F, triR, uE, EF, C);

    std::vector<int> sorted_indices;
    math::argsort_array(C.data(), C.size(), sorted_indices);

#if !defined(NDEBUG)
    assert(sorted_indices.size() == C.size());
    for (int ii = 0; ii < sorted_indices.size()-1; ++ii) {
        const int i0 = sorted_indices[ii+0];
        const int i1 = sorted_indices[ii+1];
        assert(C[i0] >= C[i1]);
    }
#endif

    const auto n_subdivision = math::clip<size_t>(size_t(double(uE.rows())*ratio), 0, uE.rows()-1);
    SE = std::vector<int>{sorted_indices.data(), sorted_indices.data()+n_subdivision};
}


template<typename IndexList>
void disjoint_edges(
        const IndexList &SE /* edge indices */,
        const Eigen::MatrixXi &uE,
        const std::unordered_set<int> &IP /* indices of pinned down vertices */,
        std::vector<int> &DSE /* disjoint edge indices */) {

    DSE.clear();
    std::unordered_set<int> ISP;

    for (const int iE: SE) {
        const int ip0 = uE(iE,0);
        const int ip1 = uE(iE,1);
        if (IP.find(ip0) != IP.end()) { continue; }
        if (IP.find(ip1) != IP.end()) { continue; }
        if (ISP.find(ip0) != ISP.end()) { continue; }
        if (ISP.find(ip1) != ISP.end()) { continue; }
        ISP.insert(ip0);
        ISP.insert(ip1);
        DSE.push_back(iE);
    }
}


} // namespace stealth