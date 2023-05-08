#include "fit_rotations_normal.h"

void fit_rotations_normal(
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXd & tarN,
	Eigen::MatrixXd & U,
    Eigen::MatrixXd & RAll,
	normal_driven_data & data)
{
    using namespace Eigen;
	using namespace std;

    int nEle = data.A.size();
    data.objValVec.setZero(nEle);

    RAll.resize(nEle*3,3);

    igl::parallel_for(
        nEle,
        [&data, &RAll, &U, &tarN](const int kk)
        {
            // get parameters
            VectorXd n = data.N.row(kk).transpose();
            VectorXd n_target = tarN.row(kk).transpose();
            MatrixXd R;

            // get energy parameters
            // Note: dVn = [dV n], dUn = [dU z-u]
            MatrixXi hE = data.E_N[kk];
            MatrixXd dU(3,hE.rows()); 
            {
                MatrixXd E0_U, E1_U;
                igl::slice(U,hE.col(0),1,E0_U);
                igl::slice(U,hE.col(1),1,E1_U);
                dU = (E1_U - E0_U).transpose();
            }

            // Note:
            // S = [dV n] * [W 0; 0 rho] * [dU (z-u)]'
            //   = dV * W * dU' + n * rho * (z-u)'
            //   = Spre + n * rho * (z-u)'
            MatrixXd dV = data.dV[kk];
            VectorXd Wvec = data.Wvec[kk];
            Matrix3d Spre = dV * Wvec.asDiagonal() * dU.transpose();

            // SVD
            MatrixXd S = Spre + data.lambda * data.A(kk) * n * n_target.transpose();
            assert(!std::isnan(S.determinant()));
            /*
             * [ Stealth Shaper ]: As igl::fit_rotations_SSE occasionally outputs NaNs,
             * we replace the rotation solver with ours using Eigen::JacobiSVD.
             */
//            igl::fit_rotations_SSE(S, R);
            stealth::fit_rotations(S, R);
            assert(!std::isnan(R.determinant()));

            // save parameters
            RAll.block(3*kk,0,3,3) = R;
            
            if (data.computeEnergy == true)
            {
                // save objective
                double objVal = 
                    ((R*dV-dU)*Wvec.asDiagonal()*(R*dV-dU).transpose()).trace()
                    + data.lambda * data.A(kk) * (R*n - n_target).transpose() * (R*n - n_target);
                data.objValVec(kk) = objVal;
            }

        }   
    ,1000);
    data.objVal = data.objValVec.sum();
}
