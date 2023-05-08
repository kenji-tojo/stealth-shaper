#pragma once

#include <memory>

#include <Eigen/Dense>


namespace stealth {

class AdaptiveMesh {
public:
    using MatrixV = Eigen::MatrixXd;
    using MatrixF = Eigen::MatrixXi;

    AdaptiveMesh(const MatrixV &V, const MatrixF &F);

    ~AdaptiveMesh();

    unsigned int split_edge(unsigned int ip0, unsigned int ip1);
    unsigned int collapse_edge(unsigned int ip0, unsigned int ip1);
    bool flip_edge(unsigned int ip0, unsigned int ip1);
    std::vector<unsigned int> cleanup();

    [[nodiscard]] MatrixV V() const;
    [[nodiscard]] MatrixF F() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;

};

} // namespace stealth