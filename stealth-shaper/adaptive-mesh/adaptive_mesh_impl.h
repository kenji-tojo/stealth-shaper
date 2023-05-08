/*
MIT License

Copyright (c) 2022 Nobuyuki Umetani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <iostream>
#include <vector>

#include <Eigen/Core>

namespace adaptive {
namespace {

class Tri {
public:
    inline unsigned operator[](int i) const {
        assert(i < 3);
        return v[i];
    }

public:
    unsigned int v[3];
    unsigned int s[3];
};

class Vtx {
public:
    Vtx() : e(UINT_MAX), d(0) {}

    Vtx(const Vtx &) = default;

    Vtx(int ielem, unsigned int idir) : e(ielem), d(idir) {}

public:
    unsigned int e;
    unsigned int d;
};

void vtx2tri(
        std::vector<unsigned int> &vtx2idx,
        std::vector<unsigned int> &idx2tri,
        const std::vector<Tri> &tris,
        const unsigned int num_vtx) {
    vtx2idx.assign(num_vtx + 1, 0);
    for (const auto &itri: tris) {
        for (unsigned int inotri: itri.v) {
            vtx2idx[inotri + 1]++;
        }
    }
    for (unsigned int ipoin = 0; ipoin < num_vtx; ipoin++) {
        vtx2idx[ipoin + 1] += vtx2idx[ipoin];
    }
    const unsigned int nelsup = vtx2idx[num_vtx];
    idx2tri.resize(nelsup);
    for (unsigned int itri = 0; itri < tris.size(); itri++) {
        for (unsigned int ipoin0: tris[itri].v) {
            const unsigned int ielsup = vtx2idx[ipoin0];
            idx2tri[ielsup] = itri;
            vtx2idx[ipoin0]++;
        }
    }
    for (unsigned int ipoin = num_vtx; ipoin > 0; ipoin--) {
        vtx2idx[ipoin] = vtx2idx[ipoin - 1];
    }
    vtx2idx[0] = 0;
}

void tri2tri(
        std::vector<Tri> &tris,
        const unsigned int num_vtx,
        const std::vector<unsigned int> &vtx2idx,
        const std::vector<unsigned int> &idx2vtx) {

    std::vector<int> tmp_poin(num_vtx, 0);
    unsigned int inpofa[2];

    const size_t nTri = tris.size();
    for (unsigned int itri = 0; itri < nTri; itri++) {
        for (unsigned int iedtri = 0; iedtri < 3; iedtri++) {
            for (unsigned int ipoed = 0; ipoed < 2; ipoed++) {
                inpofa[ipoed] = tris[itri].v[(iedtri + 1 + ipoed) % 3];
                tmp_poin[inpofa[ipoed]] = 1;
            }
            const unsigned int ipoin0 = inpofa[0];
            bool iflg = false;
            for (unsigned int ielsup = vtx2idx[ipoin0]; ielsup < vtx2idx[ipoin0 + 1]; ielsup++) {
                const unsigned int jtri0 = idx2vtx[ielsup];
                if (jtri0 == itri) continue;
                for (unsigned int jedtri = 0; jedtri < 3; jedtri++) {
                    iflg = true;
                    for (unsigned int jpoed = 0; jpoed < 2; jpoed++) {
                        const unsigned int jpoin0 = tris[jtri0].v[(jedtri + 1 + jpoed) % 3];
                        if (tmp_poin[jpoin0] == 0) {
                            iflg = false;
                            break;
                        }
                    }
                    if (iflg) {
                        tris[itri].s[iedtri] = jtri0;
                        break;
                    }
                }
                if (iflg) break;
            }
            if (!iflg) {
                tris[itri].s[iedtri] = UINT_MAX;
            }
            for (unsigned int ipofa: inpofa) {
                tmp_poin[ipofa] = 0;
            }
        }
    }
}

void initialize(
        std::vector<Vtx> &vtxs,
        std::vector<Tri> &tris,
        //
        const unsigned int *tri2vtx,
        const size_t num_tri,
        const size_t num_vtx) {
    vtxs.resize(num_vtx);
    for (unsigned int ipo = 0; ipo < num_vtx; ++ipo) {
        vtxs[ipo].e = UINT_MAX; // for unreffered point
        vtxs[ipo].d = 0;
    }
    tris.resize(num_tri);
    for (unsigned int itri = 0; itri < num_tri; itri++) {
        tris[itri].v[0] = tri2vtx[itri * 3 + 0];
        tris[itri].v[1] = tri2vtx[itri * 3 + 1];
        tris[itri].v[2] = tri2vtx[itri * 3 + 2];
    }
    for (unsigned int itri = 0; itri < num_tri; itri++) {
        const unsigned int i1 = tris[itri].v[0];
        const unsigned int i2 = tris[itri].v[1];
        const unsigned int i3 = tris[itri].v[2];
        vtxs[i1].e = itri;
        vtxs[i1].d = 0;
        vtxs[i2].e = itri;
        vtxs[i2].d = 1;
        vtxs[i3].e = itri;
        vtxs[i3].d = 2;
    }
    {
        std::vector<unsigned int> vtx2idx, idx2tri;
        vtx2tri(vtx2idx, idx2tri,
                tris, vtxs.size());
        tri2tri(tris,
                vtxs.size(), vtx2idx, idx2tri);
    }
}

unsigned int adjacent_edge_idx(
        const Tri &t0,
        unsigned int ied0,
        const std::vector<Tri> &tris) {
    const unsigned int iv0 = t0.v[(ied0 + 1) % 3];
    const unsigned int iv1 = t0.v[(ied0 + 2) % 3];
    assert(iv0 != iv1);
    const unsigned int it1 = t0.s[ied0];
    assert(it1 != UINT_MAX);
    if (tris[it1].v[1] == iv1 && tris[it1].v[2] == iv0) { return 0; }
    if (tris[it1].v[2] == iv1 && tris[it1].v[0] == iv0) { return 1; }
    if (tris[it1].v[0] == iv1 && tris[it1].v[1] == iv0) { return 2; }
    assert(false);
}

bool insert_point_to_edge(
        const unsigned int ipo_ins,    //the index of the new point
        const unsigned int itri_ins,  //triangle index
        const unsigned int ied_ins,  //edge index
        std::vector<Vtx> &vtxs,
        std::vector<Tri> &tris) {
    assert(itri_ins < tris.size());
    assert(ipo_ins < vtxs.size());
    assert(tris[itri_ins].s[ied_ins] != UINT_MAX);

    const unsigned int itri_adj = tris[itri_ins].s[ied_ins];
    const unsigned int ied_adj = adjacent_edge_idx(tris[itri_ins], ied_ins, tris);
    assert(itri_adj < tris.size() && ied_ins < 3);

    const unsigned int itri0 = itri_ins;
    const unsigned int itri1 = itri_adj;
    const auto itri2 = static_cast<unsigned int>(tris.size());
    const auto itri3 = static_cast<unsigned int>(tris.size() + 1);

    tris.resize(tris.size() + 2);

    const Tri oldA = tris[itri_ins];
    const Tri oldB = tris[itri_adj];

    const unsigned int inoA0 = ied_ins;
    const unsigned int inoA1 = (ied_ins + 1) % 3;
    const unsigned int inoA2 = (ied_ins + 2) % 3;

    const unsigned int inoB0 = ied_adj;
    const unsigned int inoB1 = (ied_adj + 1) % 3;
    const unsigned int inoB2 = (ied_adj + 2) % 3;

    assert(oldA.v[inoA1] == oldB.v[inoB2]);
    assert(oldA.v[inoA2] == oldB.v[inoB1]);
    assert(oldA.s[inoA0] == itri1);
    assert(oldB.s[inoB0] == itri0);

    vtxs[ipo_ins].e = itri0;
    vtxs[ipo_ins].d = 0;
    vtxs[oldA.v[inoA2]].e = itri0;
    vtxs[oldA.v[inoA2]].d = 1;
    vtxs[oldA.v[inoA0]].e = itri1;
    vtxs[oldA.v[inoA0]].d = 1;
    vtxs[oldB.v[inoB2]].e = itri2;
    vtxs[oldB.v[inoB2]].d = 1;
    vtxs[oldB.v[inoB0]].e = itri3;
    vtxs[oldB.v[inoB0]].d = 1;

    {
        Tri &tri0 = tris[itri0];
        tri0.v[0] = ipo_ins;
        tri0.v[1] = oldA.v[inoA2];
        tri0.v[2] = oldA.v[inoA0];
        tri0.s[0] = oldA.s[inoA1];
        tri0.s[1] = itri1;
        tri0.s[2] = itri3;
    }
    if (oldA.s[inoA1] != UINT_MAX) {
        const unsigned int jt0 = oldA.s[inoA1];
        assert(jt0 < tris.size());
        const unsigned int jno0 = adjacent_edge_idx(oldA, inoA1, tris);
        tris[jt0].s[jno0] = itri0;
    }

    {
        Tri &tri1 = tris[itri1];
        tri1.v[0] = ipo_ins;
        tri1.v[1] = oldA.v[inoA0];
        tri1.v[2] = oldA.v[inoA1];
        tri1.s[0] = oldA.s[inoA2];
        tri1.s[1] = itri2;
        tri1.s[2] = itri0;
    }
    if (oldA.s[inoA2] != UINT_MAX) {
        const unsigned int jt0 = oldA.s[inoA2];
        assert(jt0 < tris.size());
        const unsigned int jno0 = adjacent_edge_idx(oldA, inoA2, tris);
        tris[jt0].s[jno0] = itri1;
    }

    {
        Tri &tri2 = tris[itri2];
        tri2.v[0] = ipo_ins;
        tri2.v[1] = oldB.v[inoB2];
        tri2.v[2] = oldB.v[inoB0];
        tri2.s[0] = oldB.s[inoB1];
        tri2.s[1] = itri3;
        tri2.s[2] = itri1;
    }
    if (oldB.s[inoB1] != UINT_MAX) {
        const unsigned int jt0 = oldB.s[inoB1];
        assert(jt0 < tris.size());
        const unsigned int jno0 = adjacent_edge_idx(oldB, inoB1, tris);
        tris[jt0].s[jno0] = itri2;
    }

    {
        Tri &tri3 = tris[itri3];
        tri3.v[0] = ipo_ins;
        tri3.v[1] = oldB.v[inoB0];
        tri3.v[2] = oldB.v[inoB1];
        tri3.s[0] = oldB.s[inoB2];
        tri3.s[1] = itri0;
        tri3.s[2] = itri2;
    }
    if (oldB.s[inoB2] != UINT_MAX) {
        const unsigned int jt0 = oldB.s[inoB2];
        assert(jt0 < tris.size());
        const unsigned int jno0 = adjacent_edge_idx(oldB, inoB2, tris);
        tris[jt0].s[jno0] = itri3;
    }
    return true;
}

bool move_ccw(
        unsigned int &itri_cur,
        unsigned int &inotri_cur,
        unsigned int itri_adj,
        const std::vector<Tri> &tris) {
    const unsigned int inotri1 = (inotri_cur + 1) % 3;
    if (tris[itri_cur].s[inotri1] == itri_adj) { return false; }
    const unsigned int itri_nex = tris[itri_cur].s[inotri1];
    assert(itri_nex < tris.size());
    const unsigned int ino2 = adjacent_edge_idx(tris[itri_cur], inotri1, tris);
    const unsigned int inotri_nex = (ino2 + 1) % 3;
    assert(tris[itri_cur].v[inotri_cur] == tris[itri_nex].v[inotri_nex]);
    itri_cur = itri_nex;
    inotri_cur = inotri_nex;
    return true;
}

bool move_cw(
        unsigned int &itri_cur,
        unsigned int &inotri_cur,
        unsigned int itri_adj,
        const std::vector<Tri> &tris) {
    const unsigned int inotri1 = (inotri_cur + 2) % 3;
    if (tris[itri_cur].s[inotri1] == itri_adj) { return false; }
    const unsigned int itri_nex = tris[itri_cur].s[inotri1];
    assert(itri_nex < tris.size());
    const unsigned int ino2 = adjacent_edge_idx(tris[itri_cur], inotri1, tris);
    const unsigned int inotri_nex = (ino2 + 2) % 3;
    assert(tris[itri_cur].v[inotri_cur] == tris[itri_nex].v[inotri_nex]);
    itri_cur = itri_nex;
    inotri_cur = inotri_nex;
    return true;
}

bool DeleteTri(
        unsigned int itri_to,
        std::vector<Vtx> &vtxs,
        std::vector<Tri> &tris) {
    if (itri_to >= tris.size()) return true;
    // -------------
    assert(tris[itri_to].s[0] == UINT_MAX);
    assert(tris[itri_to].s[1] == UINT_MAX);
    assert(tris[itri_to].s[2] == UINT_MAX);
    assert(!tris.empty());
    const size_t itri_from = tris.size() - 1;
    if (itri_to == itri_from) {
        tris.resize(tris.size() - 1);
        return true;
    }
    tris[itri_to] = tris[itri_from];
    tris.resize(tris.size() - 1);
    for (int iedtri = 0; iedtri < 3; iedtri++) {
        if (tris[itri_to].s[iedtri] == UINT_MAX) continue;
        const unsigned int itri_adj = tris[itri_to].s[iedtri];
        const unsigned int iedtri_adj = adjacent_edge_idx(tris[itri_to], iedtri, tris);
        assert(itri_adj < tris.size());
        assert(tris[itri_adj].s[iedtri_adj] == itri_from);
        tris[itri_adj].s[iedtri_adj] = itri_to;
    }
    for (unsigned int inotri = 0; inotri < 3; inotri++) {
        const unsigned int ipo0 = tris[itri_to].v[inotri];
        vtxs[ipo0].e = itri_to;
        vtxs[ipo0].d = inotri;
    }
    return true;
}

bool collapse_edge(
        const unsigned int itri_del,
        const unsigned int ied_del,
        std::vector<Vtx> &vtxs,
        std::vector<Tri> &tris) {
    assert(itri_del < tris.size() && ied_del < 3);
    if (tris[itri_del].s[ied_del] == UINT_MAX) {
        std::cout << "Error!-->Not Implemented: Mesh with hole" << std::endl;
        assert(0);
    }

    const unsigned int itri_adj = tris[itri_del].s[ied_del];
    const unsigned int ied_adj = adjacent_edge_idx(tris[itri_del], ied_del, tris);
    assert(itri_adj < tris.size() && ied_adj < 3);
    assert(tris[itri_adj].s[ied_adj] == itri_del);

    // do nothing and return false if the collapsing edge is on the boundary
    if (tris[itri_del].s[(ied_del + 1) % 3] == UINT_MAX) return false;
    if (tris[itri_del].s[(ied_del + 2) % 3] == UINT_MAX) return false;
    if (tris[itri_adj].s[(ied_adj + 1) % 3] == UINT_MAX) return false;
    if (tris[itri_adj].s[(ied_adj + 2) % 3] == UINT_MAX) return false;

    const unsigned int itA = itri_del;
    const unsigned int itB = itri_adj;
    const unsigned int itC = tris[itA].s[(ied_del + 1) % 3];
    const unsigned int itD = tris[itA].s[(ied_del + 2) % 3];
    const unsigned int itE = tris[itB].s[(ied_adj + 1) % 3];
    const unsigned int itF = tris[itB].s[(ied_adj + 2) % 3];

    const unsigned int inoA0 = ied_del;
    const unsigned int inoA1 = (inoA0 + 1) % 3;
    const unsigned int inoA2 = (inoA0 + 2) % 3; // point to be deleted

    const unsigned int inoB0 = ied_adj;
    const unsigned int inoB1 = (inoB0 + 1) % 3; // point to be deleted
    const unsigned int inoB2 = (inoB0 + 2) % 3;

    const unsigned int inoC0 = adjacent_edge_idx(tris[itA], inoA1, tris);
    const unsigned int inoC1 = (inoC0 + 1) % 3;
    const unsigned int inoC2 = (inoC0 + 2) % 3;

    const unsigned int inoD0 = adjacent_edge_idx(tris[itA], inoA2, tris);
    const unsigned int inoD1 = (inoD0 + 1) % 3;

    const unsigned int inoE0 = adjacent_edge_idx(tris[itB], inoB1, tris);
    const unsigned int inoE1 = (inoE0 + 1) % 3;
    const unsigned int inoE2 = (inoE0 + 2) % 3;

    const unsigned int inoF0 = adjacent_edge_idx(tris[itB], inoB2, tris);
    const unsigned int inoF1 = (inoF0 + 1) % 3;

    if (tris[itC].s[inoC2] == itD) { // additinoal two triangles to be deleted
        assert(tris[itD].s[inoD1] == itC);
        // TODO: implement this collapse
        return false;
    }

    if (tris[itE].s[inoE2] == itF) { // additinoal two triangles to be deleted
        assert(tris[itF].s[inoF1] == itE);
        // TODO: implement this collapse
        return false;
    }

    if (itC == itF && itD == itE) return false;

    const Tri oldA = tris[itA];
    const Tri oldB = tris[itB];

    const unsigned int ipoW = oldA.v[inoA0];
    const unsigned int ipoX = oldA.v[inoA1];
    const unsigned int ipoY = oldB.v[inoB0];
    const unsigned int ipoZ = oldB.v[inoB1];  // point to be deleted

    assert(tris[itD].v[inoD1] == ipoX);
    assert(tris[itF].v[inoF1] == ipoZ);

    { // check if elements around ipX and elements around ipZ share common triangles
        std::vector<unsigned int> ring1;
        { // set triangle index from point 0 to point 1
            unsigned int jtri = itF;
            unsigned int jnoel_c = inoF1;
            for (;;) {
                assert(jtri < tris.size() && jnoel_c < 3 && tris[jtri].v[jnoel_c] == ipoZ);
                ring1.push_back(tris[jtri].v[(jnoel_c + 2) % 3]);
                if (!move_ccw(jtri, jnoel_c, UINT_MAX, tris)) { return false; }
                if (jtri == itC) break;
            }
        }
        std::vector<unsigned int> ring2;
        { // set triangle index from point 0 to point 1
            unsigned int jtri = itD;
            unsigned int jnoel_c = inoD1;
            for (;;) {
                assert(jtri < tris.size() && jnoel_c < 3 && tris[jtri].v[jnoel_c] == ipoX);
                ring2.push_back(tris[jtri].v[(jnoel_c + 2) % 3]);
                if (!move_ccw(jtri, jnoel_c, UINT_MAX, tris)) { return false; }
                if (jtri == itE) break;
            }
        }
        sort(ring1.begin(), ring1.end());
        sort(ring2.begin(), ring2.end());
        std::vector<unsigned int> insc(ring1.size());
        auto it = set_intersection(ring1.begin(), ring1.end(),
                                   ring2.begin(), ring2.end(),
                                   insc.begin());
        if (it != insc.begin()) { return false; } // ring1 and ring2 have intersection
    }

    assert(oldA.v[inoA1] == oldB.v[inoB2]);
    assert(oldA.v[inoA2] == oldB.v[inoB1]);
    assert(oldA.s[inoA0] == itB);
    assert(oldB.s[inoB0] == itA);

    // ---------------------------------
    // change from there

    vtxs[ipoW].e = itC;
    vtxs[ipoW].d = inoC1;
    vtxs[ipoY].e = itE;
    vtxs[ipoY].d = inoE1;
    vtxs[ipoX].e = itD;
    vtxs[ipoX].d = inoD1;
    vtxs[ipoZ].e = UINT_MAX;

    tris[itC].s[inoC0] = oldA.s[inoA2];
    if (oldA.s[inoA2] != UINT_MAX) {
        assert(oldA.s[inoA2] < tris.size());
        tris[itD].s[inoD0] = itC;
    }

    tris[itD].s[inoD0] = oldA.s[inoA1];
    if (oldA.s[inoA1] != UINT_MAX) {
        assert(oldA.s[inoA1] < tris.size());
        tris[itC].s[inoC0] = itD;
    }

    tris[itE].s[inoE0] = oldB.s[inoB2];
    if (oldB.s[inoB2] != UINT_MAX) {
        assert(oldB.s[inoB2] < tris.size());
        tris[itF].s[inoF0] = itE;
    }

    tris[itF].s[inoF0] = oldB.s[inoB1];
    if (oldB.s[inoB1] != UINT_MAX) {
        assert(oldB.s[inoB1] < tris.size());
        tris[itE].s[inoE0] = itF;
    }

    { // set triangle vtx index from ipoZ to ipoX
        std::vector<std::pair<unsigned int, unsigned int> > aItIn; // itri, inode
        unsigned int jtri = itF;
        unsigned int jnoel_c = inoF1;
        for (;;) { // MoveCCW cannot be used here
            aItIn.emplace_back(jtri, jnoel_c);
            assert(jtri < tris.size() && jnoel_c < 3 && tris[jtri].v[jnoel_c] == ipoZ);
            if (!move_ccw(jtri, jnoel_c, itD, tris)) { break; }
        }
        for (auto &itr: aItIn) {
            const unsigned int it0 = itr.first;
            const unsigned int in0 = itr.second;
            assert(tris[it0].v[in0] == ipoZ);
            tris[it0].v[in0] = ipoX;
        }
    }

    {  // isolate two triangles to be deleted
        tris[itA].s[0] = UINT_MAX;
        tris[itA].s[1] = UINT_MAX;
        tris[itA].s[2] = UINT_MAX;
        tris[itB].s[0] = UINT_MAX;
        tris[itB].s[1] = UINT_MAX;
        tris[itB].s[2] = UINT_MAX;
        const unsigned int itri_1st = (itA > itB) ? itA : itB;
        const unsigned int itri_2nd = (itA < itB) ? itA : itB;
        DeleteTri(itri_1st, vtxs, tris);
        DeleteTri(itri_2nd, vtxs, tris);
    }
    return true;
}

bool find_edge_from_two_points(
        unsigned int &itri0,
        unsigned int &inotri0,
        unsigned int &inotri1,
        //
        const unsigned int ipo0,
        const unsigned int ipo1,
        const std::vector<Vtx> &vtxs,
        const std::vector<Tri> &tris) {
    if (vtxs[ipo0].e == UINT_MAX || vtxs[ipo1].e == UINT_MAX)
        return false;
    unsigned int itc = vtxs[ipo0].e;
    unsigned int inc = vtxs[ipo0].d;
    for (;;) {  // serch clock-wise
        assert(tris[itc].v[inc] == ipo0);
        const unsigned int inotri2 = (inc + 1) % 3;
        if (tris[itc].v[inotri2] == ipo1) {
            itri0 = itc;
            inotri0 = inc;
            inotri1 = inotri2;
            assert(tris[itri0].v[inotri0] == ipo0);
            assert(tris[itri0].v[inotri1] == ipo1);
            return true;
        }
        if (!move_cw(itc, inc, UINT_MAX, tris)) { break; }
        if (itc == vtxs[ipo0].e) return false;
    }
    // -------------
    inc = vtxs[ipo0].d;
    itc = vtxs[ipo0].e;
    for (;;) { // search counter clock-wise
        assert(tris[itc].v[inc] == ipo0);
        if (!move_ccw(itc, inc, UINT_MAX, tris)) { break; }
        if (itc == vtxs[ipo0].e) {  // end if it goes around
            itri0 = 0;
            inotri0 = 0;
            inotri1 = 0;
            return false;
        }
        const unsigned int inotri2 = (inc + 1) % 3;
        if (tris[itc].v[inotri2] == ipo1) {
            itri0 = itc;
            inotri0 = inc;
            inotri1 = inotri2;
            assert(tris[itri0].v[inotri0] == ipo0);
            assert(tris[itri0].v[inotri1] == ipo1);
            return true;
        }
    }
    return false;
}

void assert_tris(
        [[maybe_unused]] const std::vector<Tri> &tris) {
#if !defined(NDEBUG)
    const size_t ntri = tris.size();
    for (unsigned int itri = 0; itri < ntri; itri++) {
        const Tri &tri = tris[itri];
        if (tri.v[0] == UINT_MAX) {
            assert(tri.v[1] == UINT_MAX);
            assert(tri.v[2] == UINT_MAX);
            continue;
        }
        assert(tri.v[0] != tri.v[1]);
        assert(tri.v[1] != tri.v[2]);
        assert(tri.v[2] != tri.v[0]);
        assert((tri.s[0] != tri.s[1]) || tri.s[0] == UINT_MAX);
        assert((tri.s[1] != tri.s[2]) || tri.s[1] == UINT_MAX);
        assert((tri.s[2] != tri.s[0]) || tri.s[0] == UINT_MAX);
        for (int iedtri = 0; iedtri < 3; iedtri++) {
            if (tri.s[iedtri] == UINT_MAX) continue;
            assert(tri.s[iedtri] < tris.size());
            const unsigned int jtri = tri.s[iedtri];
            assert(jtri < ntri);
            const unsigned int jno = adjacent_edge_idx(tris[itri], iedtri, tris);
            assert(tris[jtri].s[jno] == itri);
            assert(tris[itri].v[(iedtri + 1) % 3] == tris[jtri].v[(jno + 2) % 3]);
            assert(tris[itri].v[(iedtri + 2) % 3] == tris[jtri].v[(jno + 1) % 3]);
        }
    }
#endif
}

void assert_mesh(
        [[maybe_unused]] const std::vector<Vtx> &vtxs,
        [[maybe_unused]] const std::vector<Tri> &tris) {
#if !defined(NDEBUG)
    const size_t npo = vtxs.size();
    const size_t ntri = tris.size();
    for (unsigned int itri = 0; itri < ntri; itri++) {
        assert(tris[itri].v[0] < npo);
        assert(tris[itri].v[0] < npo);
        assert(tris[itri].v[0] < npo);
    }
    for (unsigned int ipoin = 0; ipoin < npo; ++ipoin) {
        const unsigned int itri0 = vtxs[ipoin].e;
        const unsigned int inoel0 = vtxs[ipoin].d;
        if (itri0 != UINT_MAX) {
            assert(itri0 < tris.size() && inoel0 < 3 && tris[itri0].v[inoel0] == ipoin);
        }
    }
#endif
}

bool flip_edge(
        unsigned int itriA,
        unsigned int ied0,
        std::vector<Vtx> &aPo,
        std::vector<Tri> &aTri) {
    assert(itriA < aTri.size() && ied0 < 3);
    if (aTri[itriA].s[ied0] == UINT_MAX) { return false; }

    const unsigned int itriB = aTri[itriA].s[ied0];
    assert(itriB < aTri.size());
    const unsigned int ied1 = adjacent_edge_idx(aTri[itriA], ied0, aTri);
    assert(ied1 < 3);
    assert(aTri[itriB].s[ied1] == itriA);

    const Tri oldA = aTri[itriA];
    const Tri oldB = aTri[itriB];

    const unsigned int noA0 = ied0;
    const unsigned int noA1 = (ied0 + 1) % 3;
    const unsigned int noA2 = (ied0 + 2) % 3;

    const unsigned int noB0 = ied1;
    const unsigned int noB1 = (ied1 + 1) % 3;
    const unsigned int noB2 = (ied1 + 2) % 3;

    assert(oldA.v[noA1] == oldB.v[noB2]);
    assert(oldA.v[noA2] == oldB.v[noB1]);

    {
        unsigned int itri, inotri0, inotri1;
        bool is_edge = find_edge_from_two_points(
                itri, inotri0, inotri1,
                oldA.v[noA0], oldB.v[noB0], aPo, aTri);
        if (is_edge) { return false; }
    }

    aPo[oldA.v[noA1]].e = itriA;
    aPo[oldA.v[noA1]].d = 0;
    aPo[oldA.v[noA0]].e = itriA;
    aPo[oldA.v[noA0]].d = 2;
    aPo[oldB.v[noB1]].e = itriB;
    aPo[oldB.v[noB1]].d = 0;
    aPo[oldB.v[noB0]].e = itriB;
    aPo[oldB.v[noB0]].d = 2;

    {
        Tri &triA = aTri[itriA];
        triA.v[0] = oldA.v[noA1];
        triA.v[1] = oldB.v[noB0];
        triA.v[2] = oldA.v[noA0];
        triA.s[0] = itriB;
        triA.s[1] = oldA.s[noA2];
        triA.s[2] = oldB.s[noB1];
    }
    if (oldA.s[noA2] != UINT_MAX) {
        const unsigned int jt0 = oldA.s[noA2];
        assert(jt0 < aTri.size() && jt0 != itriB && jt0 != itriA);
        const unsigned int jno0 = adjacent_edge_idx(oldA, noA2, aTri);
        aTri[jt0].s[jno0] = itriA;
    }
    if (oldB.s[noB1] != UINT_MAX) {
        const unsigned int jt0 = oldB.s[noB1];
        assert(jt0 < aTri.size() && jt0 != itriB && jt0 != itriA);
        const unsigned int jno0 = adjacent_edge_idx(oldB, noB1, aTri);
        aTri[jt0].s[jno0] = itriA;
    }

    {
        Tri &triB = aTri[itriB];
        triB.v[0] = oldB.v[noB1];
        triB.v[1] = oldA.v[noA0];
        triB.v[2] = oldB.v[noB0];
        triB.s[0] = (int) itriA;
        triB.s[1] = oldB.s[noB2];
        triB.s[2] = oldA.s[noA1];
    }
    if (oldB.s[noB2] != UINT_MAX) {
        const unsigned int jt0 = oldB.s[noB2];
        assert(jt0 < aTri.size());
        const unsigned int jno0 = adjacent_edge_idx(oldB, noB2, aTri);
        aTri[jt0].s[jno0] = itriB;
    }
    if (oldA.s[noA1] != UINT_MAX) {
        const unsigned int jt0 = oldA.s[noA1];
        assert(jt0 < aTri.size());
        const unsigned int jno0 = adjacent_edge_idx(oldA, noA1, aTri);
        aTri[jt0].s[jno0] = itriB;
    }
    return true;
}


// --------------------

class Mesh {
public:
    Mesh(
            std::vector<unsigned int> &tri2vtx,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &vtx2xyz) {
        vecs.resize(vtx2xyz.rows());
        for (int ipo = 0; ipo < vtx2xyz.rows(); ipo++) {
            vecs[ipo] = vtx2xyz.row(ipo);
        }
        initialize(vtxs, tris,
                   tri2vtx.data(), tri2vtx.size() / 3, vtx2xyz.size() / 3);
        assert_tris(tris);
        assert_mesh(vtxs, tris);
    }

    unsigned int split_edge(unsigned int ip0, unsigned int ip1) {
        unsigned int itri, inotri0, inotri1;
        bool is_edge = find_edge_from_two_points(
                itri, inotri0, inotri1,
                ip0, ip1, vtxs, tris);
        if (!is_edge) { return UINT_MAX; }
        unsigned int iedge = 3 - inotri0 - inotri1;
        unsigned int ipo_ins = vtxs.size();
        vtxs.resize(ipo_ins + 1);
        insert_point_to_edge(ipo_ins, itri, iedge, vtxs, tris);
        vecs.emplace_back(((vecs[ip0] + vecs[ip1])) * 0.5);
        assert_tris(tris);
        assert_mesh(vtxs, tris);
        return ipo_ins;
    }

    /**
     * @param ip0 point index.
     * @param ip1 point index. This point will be deleted
     * @return true => if the edge was collapsed, false => edge cannot be collapsed. nothing happens to the mesh
     */
    unsigned int collapse_edge(unsigned int ip0, unsigned int ip1) {
        assert(vtxs.size() == vecs.size());
        unsigned int itri, inotri0, inotri1;
        bool is_edge = find_edge_from_two_points(
                itri, inotri0, inotri1,
                ip0, ip1, vtxs, tris);
        if (!is_edge) { return false; }
        unsigned int iedge = 3 - inotri0 - inotri1;
        if (::adaptive::collapse_edge(itri, iedge, vtxs, tris)) {
            assert(vtxs[ip1].e == UINT_MAX);
            vecs[ip0] = (vecs[ip0] + vecs[ip1]) * 0.5;
            assert_tris(tris);
            assert_mesh(vtxs, tris);
            return true;
        }
        return false;
    }

    bool flip_edge(unsigned int ip0, unsigned int ip1) {
        assert(vtxs.size() == vecs.size());
        unsigned int itri, inotri0, inotri1;
        bool is_edge = find_edge_from_two_points(
                itri, inotri0, inotri1,
                ip0, ip1, vtxs, tris);
        if (!is_edge) { return false; }
        unsigned int iedge = 3 - inotri0 - inotri1;
        adaptive::flip_edge(itri, iedge, vtxs, tris);
        assert_tris(tris);
        assert_mesh(vtxs, tris);
        return true;
    }

    /**
     * remove unreferenced vertex
     * @return mapping between old vertex ID to new vertex ID
     */
    std::vector<unsigned int> cleanup() {
        assert(vtxs.size() == vecs.size());
        std::vector<unsigned int> old2new(vtxs.size(), UINT_MAX);
        unsigned int num_vtx_new = 0;
        for (unsigned int iv_old = 0; iv_old < vtxs.size(); ++iv_old) {
            if (vtxs[iv_old].e == UINT_MAX) { continue; }
            old2new[iv_old] = num_vtx_new;
            num_vtx_new += 1;
        }
        std::vector<Vtx> vtxs_new(num_vtx_new);
        std::vector<Eigen::Vector3d> vecs_new(num_vtx_new);
        for (unsigned int iv_old = 0; iv_old < vtxs.size(); ++iv_old) {
            if (old2new[iv_old] == UINT_MAX) { continue; }
            unsigned int iv_new = old2new[iv_old];
            vtxs_new[iv_new] = vtxs[iv_old];
            vecs_new[iv_new] = vecs[iv_old];
        }
        for (auto &tri: tris) {
            tri.v[0] = old2new[tri.v[0]];
            tri.v[1] = old2new[tri.v[1]];
            tri.v[2] = old2new[tri.v[2]];
        }
        vtxs = vtxs_new;
        vecs = vecs_new;
        assert_tris(tris);
        assert_mesh(vtxs_new, tris);
        return old2new;
    }

    [[nodiscard]] std::vector<uint32_t> F() const {
        std::vector<uint32_t> res(tris.size() * 3);
        for (unsigned int it = 0; it < tris.size(); ++it) {
            res[it * 3 + 0] = tris[it].v[0];
            res[it * 3 + 1] = tris[it].v[1];
            res[it * 3 + 2] = tris[it].v[2];
        }
        return res;
    }

    [[nodiscard]] Eigen::Matrix<double, -1, 3, Eigen::RowMajor> V() const {
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> v(vecs.size(), 3);
        for (unsigned int iv = 0; iv < vecs.size(); ++iv) {
            v(iv, 0) = vecs[iv].x();
            v(iv, 1) = vecs[iv].y();
            v(iv, 2) = vecs[iv].z();
        }
        return v;
    }

private:
    std::vector<Vtx> vtxs;
    std::vector<Tri> tris;
    std::vector<Eigen::Vector3d> vecs;
};

} // namespace
} // namespace adaptive


