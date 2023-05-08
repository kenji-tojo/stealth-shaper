#pragma once

#include <iostream>
#include <vector>

#include "internal/math.hpp"

#include <Eigen/Sparse>
#include <igl/edge_flaps.h>


namespace stealth {


template<typename Scalar_ = double>
class LargeSteps {
public:
    using Scalar = Scalar_;

    int p = 1; // 1: single-laplacian, 2: bi-laplacian

    LargeSteps(
            const int nV /* number of vertices */,
            const Eigen::MatrixXi &F,
            const double lambda = 0.05) {

        using namespace Eigen;

        assert(lambda >= 0. && lambda <= 1.);

        MatrixXi uE, EF, EI;
        VectorXi EMAP;
        igl::edge_flaps(F, uE, EMAP, EF, EI);
        std::cout << "LargeSteps: found " << uE.rows() << " unique edges" << std::endl;

        std::vector<Triplet<Scalar, int>> lap;
        for (int jj = 0; jj < uE.rows(); ++jj) {
            const int ip0 = uE(jj,0);
            const int ip1 = uE(jj,1);
            lap.emplace_back(ip0, ip1, -1.0);
            lap.emplace_back(ip1, ip0, -1.0);
            lap.emplace_back(ip0, ip0, 1.0);
            lap.emplace_back(ip1, ip1, 1.0);
        }

        SparseMatrix<Scalar> L(nV,nV), I(nV,nV);

        L.setFromTriplets(lap.begin(),  lap.end());
        I.setIdentity();

        const auto A = (1.0 - lambda) * L + lambda * I;
        solver.analyzePattern(A);
        solver.factorize(A);
    }


    template<typename DerivedV>
    DerivedV precondition(const Eigen::MatrixBase<DerivedV> &dV /* #V by 3 list of vertex position gradients */)  {

        using namespace std;

        if (p == 1) {
            cout << "LargeSteps::precondition: using single-laplacian" << endl;
            return solver.solve(dV);
        }
        else if (p == 2) {
            cout << "LargeSteps::precondition: using bi-laplacian" << endl;
            return solver.solve(solver.solve(dV));
        }

        assert(false);
        cout << "LargeSteps::precondition [ Warning ]: unsupported preconditioning type, "
             << "falling back to single-laplacian" << endl;
        return solver.solve(dV);
    }

private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> solver;

};

} // namespace stealth