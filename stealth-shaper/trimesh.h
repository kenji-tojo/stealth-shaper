#pragma once

#include <memory>
#include <string>

#include "internal/math.hpp"
#include "object.hpp"
#include "tensor.hpp"
#include "normal.hpp"


namespace stealth {

class TriMesh: public Object {
public:
    using Vector3 = Eigen::Vector3f;
    using MatrixV = Eigen::MatrixXf;
    using TensorV = Tensor<MatrixV>;
    using MatrixN = Eigen::MatrixXf;
    using MatrixF = Eigen::MatrixXi;
    using TensorNXY = TensorUnit3XY<MatrixN, Vector3>;
    using GradAccV = GradientAccumulatorVertex<MatrixV, MatrixF, MatrixN, Vector3>;
    using GradAccN = GradientAccumulatorNormal<MatrixN, Vector3>;

    explicit TriMesh(std::string path);
    TriMesh(Eigen::MatrixXd &&V, Eigen::MatrixXi &&F);
    TriMesh(Eigen::MatrixXf &&V, Eigen::MatrixXi &&F);

    ~TriMesh();

    void print_info() const;
    void write_obj(std::string path) const;
    void write_obj_reference(std::string path) const;

    void raycast(const Ray &ray, Hit &hit) const override;

    Eigen::MatrixXd refV;
    Eigen::MatrixXd refTriN;

    Eigen::VectorXf A; // normalized face areas
    float totalA = 0;

    float avg_edge_length = 0;

    std::shared_ptr<MatrixV> V_shared;
    std::shared_ptr<MatrixF> F_shared;
    std::shared_ptr<MatrixN> triN_shared;

#define DEFINE_DEREF(name) \
    [[nodiscard]] const typename decltype(name##_shared)::element_type &name() const { assert(name##_shared); return *name##_shared; } \
    typename decltype(name##_shared)::element_type &name() { assert(name##_shared); return *name##_shared; }

    DEFINE_DEREF(V);
    DEFINE_DEREF(F);
    DEFINE_DEREF(triN);

#undef DEFINE_DEREF

    void update_positions();

    void reset_positions();

    template<typename AdaptiveMesh, typename IndexType>
    void split_edges(
            const Eigen::MatrixXi &uE,
            const std::vector<int> &SE,
            std::vector<IndexType> &insP,
            std::vector<IndexType> &VMAP);

private:
    class Bvh;
    std::unique_ptr<Bvh> bvh;

    static void per_face_normals(
            const Eigen::MatrixXd &V,
            const Eigen::MatrixXi &F,
            Eigen::MatrixXd &triN);

};


template<typename AdaptiveMesh, typename IndexType>
void TriMesh::split_edges(
        const Eigen::MatrixXi &uE,
        const std::vector<int> &SE,
        std::vector<IndexType> &insP,
        std::vector<IndexType> &VMAP) {

    using namespace Eigen;

    const int nVold =  V().rows();

    MatrixXd refVnew;
    MatrixXi Fnew;

    insP.clear();

    // updating the reference positions
    auto adaptive_mesh = std::make_unique<AdaptiveMesh>(refV, F());
    for (const int jj: SE) {
        const int ip0 = uE(jj,0);
        const int ip1 = uE(jj,1);
        insP.push_back(adaptive_mesh->split_edge(ip0, ip1));
    }
    if constexpr(std::is_same_v<IndexType, unsigned int>) {
        VMAP = adaptive_mesh->cleanup();
    }
    else {
        auto _vm = adaptive_mesh->cleanup();
        VMAP = std::vector<IndexType>{_vm.begin(), _vm.end()};
    }
    refVnew = adaptive_mesh->V();
    Fnew = adaptive_mesh->F();


    // updating the deformed positions by repeating the same operation
    adaptive_mesh = std::make_unique<AdaptiveMesh>(V().template cast<double>(), F());
    int iIns = 0;
    for (const int jj: SE) {
        const int ip0 = uE(jj,0);
        const int ip1 = uE(jj,1);
        const auto ins = adaptive_mesh->split_edge(ip0, ip1);
        assert(insP[iIns] == ins);
        iIns += 1;
    }
#if !defined(NDEBUG)
    {
        const auto vm = adaptive_mesh->cleanup();
        const auto f = adaptive_mesh->F();
        assert(vm.size() == VMAP.size());
        for (int ii = 0; ii < vm.size(); ++ii) {
            assert(vm[ii] == VMAP[ii]);
        }
        assert(f.rows() == Fnew.rows());
        for (int kk = 0; kk < f.rows(); ++kk) {
            assert(f(kk,0) == Fnew(kk,0));
            assert(f(kk,1) == Fnew(kk,1));
            assert(f(kk,2) == Fnew(kk,2));
        }
    }
#else
    adaptive_mesh->cleanup();
#endif

    refV = refVnew;
    F() = Fnew;
    TriMesh::per_face_normals(refV, F(), refTriN);
    V() = adaptive_mesh->V().template cast<float>();
    update_positions();

    assert(insP.size() == SE.size());
    assert(VMAP.size() == nVold + SE.size());
#if !defined(NDEBUG)
    for (int ii = 0; ii < VMAP.size(); ++ii) {
        assert(VMAP[ii] == ii);
    }
    for (int ii = 0; ii < SE.size(); ++ii) {
        assert(insP[ii] == nVold+ii);
        assert(VMAP[insP[ii]] == insP[ii]);
    }
#endif
}


} // namespace stealth