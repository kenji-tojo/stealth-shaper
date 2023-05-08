#include "trimesh.h"

#include <iostream>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/per_face_normals.h>
#include <igl/avg_edge_length.h>

#include "internal/bvh.hpp"


namespace stealth {

using BvhImpl = typename internal::Bvh</*should_permute_=*/true>;

class TriMesh::Bvh: public BvhImpl {
public:
    Bvh(const Eigen::MatrixXf &V, const Eigen::MatrixXi &F): BvhImpl(V,F) {}
};

TriMesh::TriMesh(std::string path): Object("mesh") {
    MatrixF tmpF;
    std::cout << "loading mesh from " << path << std::endl;
    igl::readOBJ(path,refV,tmpF);
    igl::per_face_normals(refV, tmpF, refTriN);

    V_shared = std::make_shared<MatrixV>(refV.cast<float>());
    F_shared = std::make_shared<MatrixF>(std::move(tmpF));

    update_positions();
}

TriMesh::~TriMesh() = default;

void TriMesh::update_positions() {
    igl::doublearea(V(), F(),  A);
    totalA = A.sum() / 2.f;
    A /= (2.f * totalA);
    assert(abs(A.sum()-1.f) < 1e-6f);

    if (!triN_shared) {
        MatrixN triN;
        igl::per_face_normals(V(), F(), triN);
        triN_shared = std::make_shared<MatrixN>(std::move(triN));
    }
    else {
        igl::per_face_normals(V(), F(), triN());
    }

    avg_edge_length = igl::avg_edge_length(V(), F());

    bvh = std::make_unique<Bvh>(V(), F());
}

void TriMesh::reset_positions() {
    V() = refV.cast<float>();
    update_positions();
}

void TriMesh::print_info() const {
    using namespace std;
    cout << "mesh with "
         << V().rows() << " vertices and "
         << F().rows() << " faces" << endl;
    cout << "total area: " << totalA << endl;
    cout << "average edge length: " << avg_edge_length << endl;
}

void TriMesh::raycast(const Ray &ray, Hit &hit) const {
    if (!this->enabled || !bvh) return;

    size_t tri_id;
    float dist;
    Eigen::Vector3f nrm;

    if (bvh->raycast(ray, tri_id, dist, nrm)) {
        if (dist >= hit.dist) return;

        hit.dist = dist;
        assert(tri_id < std::numeric_limits<int>::max());
        hit.prim_id = int(tri_id);
        hit.nrm = triN().row(hit.prim_id);
//        hit.nrm = nrm;
//        assert((hit.nrm-triN().row(hit.prim_id).transpose()).norm() < 1e-5);
        hit.pos = ray.org + dist* ray.dir + 1e-4f * nrm;
        hit.wo = -ray.dir;
        hit.obj_id = this->obj_id;
        hit.mat_id = this->mat_id;
    }
}


void TriMesh::write_obj(std::string path) const {
    igl::writeOBJ(path, V(), F());
}


void TriMesh::write_obj_reference(std::string path) const {
    igl::writeOBJ(path, refV, F());
}


TriMesh::TriMesh(Eigen::MatrixXd &&V, Eigen::MatrixXi &&F): Object("mesh") {
    igl::per_face_normals(V, F, refTriN);
    refV = std::move(V);
    V_shared = std::make_shared<MatrixV>(refV.cast<float>());
    F_shared = std::make_shared<MatrixF>(std::move(F));
    update_positions();
}

TriMesh::TriMesh(Eigen::MatrixXf &&V, Eigen::MatrixXi &&F): Object("mesh") {
    refV = V.template cast<double>();
    igl::per_face_normals(refV, F, refTriN);
    V_shared = std::make_shared<MatrixV>(std::move(V));
    F_shared = std::make_shared<MatrixF>(std::move(F));
    update_positions();
}


void TriMesh::per_face_normals(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        Eigen::MatrixXd &triN) {

    igl::per_face_normals(V, F, triN);
}

} // namespace stealth