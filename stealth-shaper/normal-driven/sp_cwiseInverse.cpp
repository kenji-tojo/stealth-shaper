#include "sp_cwiseInverse.h"

void sp_cwiseInverse(
	const Eigen::SparseMatrix<double> & spM,
	Eigen::SparseMatrix<double> & spMCwiseInv)
{
	using namespace Eigen;
	spMCwiseInv.resize(spM.rows(), spM.cols());
	spMCwiseInv = spM;
	for (int k=0; k<spMCwiseInv.outerSize(); ++k) 
		for (SparseMatrix<double>::InnerIterator it(spMCwiseInv,k); it; ++it){
			it.valueRef() = 1 / (it.valueRef()+1e-8);
		}
}