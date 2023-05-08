#include "adaptive_mesh.h"

#include <algorithm>

#include "adaptive_mesh_impl.h"


namespace stealth {

struct AdaptiveMesh::Impl: public adaptive::Mesh {
public:
    using MatrixV = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;
    Impl(std::vector<unsigned int> &tri2vtx, MatrixV &vtx2xyz)
            : adaptive::Mesh(tri2vtx, vtx2xyz) {}
};

AdaptiveMesh::AdaptiveMesh(const MatrixV &V, const MatrixF &F) {

    Impl::MatrixV vtx2xyz = V; // NOTE: MatrixV != Impl::MatrixV
    std::vector<unsigned int> tri2vtx(/*n=*/F.rows()*3);
    for (int kk = 0; kk < F.rows(); ++kk) {
        tri2vtx[kk*3+0] = F(kk,0);
        tri2vtx[kk*3+1] = F(kk,1);
        tri2vtx[kk*3+2] = F(kk,2);
    }
    impl = std::make_unique<Impl>(tri2vtx, vtx2xyz);
}

AdaptiveMesh::~AdaptiveMesh() = default;


unsigned int AdaptiveMesh::split_edge(unsigned int ip0, unsigned int ip1) {
    if (ip0 > ip1) { std::swap(ip0, ip1); }
    return impl->split_edge(ip0, ip1);
}

unsigned int AdaptiveMesh::collapse_edge(unsigned int ip0, unsigned int ip1) {
    if (ip0 > ip1) { std::swap(ip0, ip1); }
    return impl->collapse_edge(ip0, ip1);
}

bool AdaptiveMesh::flip_edge(unsigned int ip0, unsigned int ip1) {
    if (ip0 > ip1) { std::swap(ip0, ip1); }
    return impl->flip_edge(ip0, ip1);
}

std::vector<unsigned int> AdaptiveMesh::cleanup() {
    return impl->cleanup();
}

AdaptiveMesh::MatrixV AdaptiveMesh::V() const {
    using namespace std;
    cout << "AdaptiveMesh::V [ Warning ]: this getter incurs copy which could harm performance" << endl;
    return impl->V();
}

AdaptiveMesh::MatrixF AdaptiveMesh::F() const {
    using namespace std;
    cout << "AdaptiveMesh::F [ Warning ]: this getter incurs copy which could harm performance" << endl;
    const auto tri2vtx = impl->F();
    MatrixF F_;
    F_.resize(tri2vtx.size()/3,3);
    for (int kk = 0; kk < F_.rows(); ++kk) {
        F_(kk,0) = tri2vtx[kk*3+0];
        F_(kk,1) = tri2vtx[kk*3+1];
        F_(kk,2) = tri2vtx[kk*3+2];
    }
    return F_;
}

} // namespace stealth