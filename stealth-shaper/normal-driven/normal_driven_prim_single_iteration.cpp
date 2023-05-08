#include "normal_driven_prim_single_iteration.h"

void normal_driven_prim_single_iteration(
	const Eigen::MatrixXd & V,
	const Eigen::MatrixXi & F,
    const Eigen::MatrixXd & tarN,
    Eigen::MatrixXd & U,
    normal_driven_data & data)
{
    using namespace Eigen;
    using namespace std;

    int nEle = data.A.size();
    int nV = U.rows();

    assert(tarN.rows() == nEle);
    assert(!data.useScale);

    // local step
    MatrixXd RAll;
    {
        PROFC_NODE("local step")
        fit_rotations_normal(F, tarN, U, RAll, data);
    }

    // global step
    MatrixXd Upre = U;
    {
        PROFC_NODE("global step")
        MatrixXd RAllT = RAll.transpose();
        Map<const VectorXd> Rcol(RAllT.data(), RAllT.size());
        VectorXd Bcol = data.K.transpose() * Rcol;
        for(int dim=0; dim<U.cols(); dim++)
        {
            VectorXd Uc,Bc,bcc;
            Bc = Bcol.block(dim*nV,0,nV,1);
            bcc = data.bc.col(dim);
            min_quad_with_fixed_solve(
                data.solver_data,Bc,bcc,VectorXd(),Uc);
            U.col(dim) = Uc;
        }
    }

    // print optimization date
    data.reldV = (U-Upre).cwiseAbs().maxCoeff() / (U-V).cwiseAbs().maxCoeff();
    // cout << "reldV:" << scientific << data.reldV << endl;
}