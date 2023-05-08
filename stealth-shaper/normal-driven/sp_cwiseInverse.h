#ifndef SP_CWISEINVERSE_H
#define SP_CWISEINVERSE_H

#include <Eigen/Sparse>

void sp_cwiseInverse(
	const Eigen::SparseMatrix<double> & spM,
	Eigen::SparseMatrix<double> & spMCwiseInv);
#endif