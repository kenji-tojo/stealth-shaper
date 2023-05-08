#pragma once

#include <cstdint>

#include <unordered_set>

#include "internal/math.hpp"


namespace stealth {

template<typename DerivedEV>
void edge_neighbors(
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXi &uE,
        const Eigen::MatrixXi &EF,
        Eigen::PlainObjectBase<DerivedEV> &EV /* #E by 4 list of neighboring vertex indices for each edge */) {

    using IndexType = typename DerivedEV::Scalar;

    EV.resize(uE.rows(), 4);

    std::unordered_set<IndexType> nei;

    for (int iE =0; iE < uE.rows(); ++iE) {
        nei.clear();

        const int k0 = EF(iE,0);
        const int k1 = EF(iE,1);

        nei.insert(F(k0,0));
        nei.insert(F(k0,1));
        nei.insert(F(k0,2));
        nei.insert(F(k1,0));
        nei.insert(F(k1,1));
        nei.insert(F(k1,2));

        assert(nei.size() == 4);

        int inei = 0;
        for (const auto ip: nei) {
            EV(iE,inei) = ip;
            inei += 1;
        }
    }
}


template<typename DerivedDE>
void edge_set_difference(
        const Eigen::MatrixXi &allE,
        const Eigen::MatrixXi &subE /* subset of the edges which may have different ordering from allE */,
        Eigen::PlainObjectBase<DerivedDE> &DE /* = allE - subE as an edge set */) {

    static_assert(std::is_same_v<typename Eigen::MatrixXi::Scalar, int32_t>);

    std::unordered_set<uint64_t> uniE;

    for (int jj = 0; jj < allE.rows(); ++jj) {
        const int e0 = allE(jj,0);
        const int e1 = allE(jj,1);
        uint64_t edge_id = 0;
        if (e0 < e1) {
            reinterpret_cast<uint32_t*>(&edge_id)[0] = e0;
            reinterpret_cast<uint32_t*>(&edge_id)[1] = e1;
        }
        else {
            reinterpret_cast<uint32_t*>(&edge_id)[0] = e1;
            reinterpret_cast<uint32_t*>(&edge_id)[1] = e0;
        }
        assert(edge_id > 0);
        uniE.insert(edge_id);
    }
    assert(uniE.size() == allE.rows());

    for (int jj = 0; jj < subE.rows(); ++jj) {
        const int e0 = subE(jj,0);
        const int e1 = subE(jj,1);
        uint64_t edge_id = 0;
        if (e0 < e1) {
            reinterpret_cast<uint32_t*>(&edge_id)[0] = e0;
            reinterpret_cast<uint32_t*>(&edge_id)[1] = e1;
        }
        else {
            reinterpret_cast<uint32_t*>(&edge_id)[0] = e1;
            reinterpret_cast<uint32_t*>(&edge_id)[1] = e0;
        }
        assert(edge_id > 0);
        uniE.erase(edge_id);
    }
    assert(uniE.size() == allE.rows()-subE.rows());

    DE.resize(allE.rows()-subE.rows(), 2);
    int jj = 0;
    for (const uint64_t edge_id: uniE) {
        DE(jj,0) = reinterpret_cast<const uint32_t*>(&edge_id)[0];
        DE(jj,1) = reinterpret_cast<const uint32_t*>(&edge_id)[1];
        jj += 1;
    }
    assert(jj == DE.rows());
}


template<typename DerivedIE>
void inserted_edges(
        const Eigen::MatrixXi &uE /* #E by 2 edge-to-vertex list */,
        const Eigen::MatrixXi &EV /* #E by 4 list of vertex indices adjacent to each edge */,
        const std::vector<int> &SE /* list of subdivided edge indices */,
        const std::vector<int> &insP /* #SE list of inserted vertex indices */,
        const std::vector<int> &VMAP /* old-to-new vertex index mapping */,
        Eigen::PlainObjectBase<DerivedIE> &IE /* #SE*4 by 2 list of newly inserted edges */) {

    assert(SE.size() == insP.size());

    IE.resize(SE.size()*4, 2);

    int ip[4];

    for (int ii = 0; ii < SE.size(); ++ii) {
        const int ise = SE[ii];

        ip[0] = ip[1] = ip[2] = ip[3] = -1;

        ip[0] = uE(ise,0);
        ip[1] = uE(ise,1);

        const int iq0 = EV(ise,0);
        const int iq1 = EV(ise,1);
        const int iq2 = EV(ise,2);
        const int iq3 = EV(ise,3);

        int jj = 2;
        if (iq0!=ip[0] && iq0!=ip[1]) { assert(jj<4); ip[jj] = iq0; jj+=1; }
        if (iq1!=ip[0] && iq1!=ip[1]) { assert(jj<4); ip[jj] = iq1; jj+=1; }
        if (iq2!=ip[0] && iq2!=ip[1]) { assert(jj<4); ip[jj] = iq2; jj+=1; }
        if (iq3!=ip[0] && iq3!=ip[1]) { assert(jj<4); ip[jj] = iq3; jj+=1; }
        assert(jj==4);

        assert(ip[0]>=0 && ip[1]>=0 && ip[2]>=0 && ip[3]>=0);

        const int ins = insP[ii];
        IE(ii*4+0, 0) = VMAP[ip[0]];
        IE(ii*4+0, 1) = VMAP[ins];
        IE(ii*4+1, 0) = VMAP[ip[1]];
        IE(ii*4+1, 1) = VMAP[ins];
        IE(ii*4+2, 0) = VMAP[ip[2]];
        IE(ii*4+2, 1) = VMAP[ins];
        IE(ii*4+3, 0) = VMAP[ip[3]];
        IE(ii*4+3, 1) = VMAP[ins];
    }
}


} // namespace stealth
