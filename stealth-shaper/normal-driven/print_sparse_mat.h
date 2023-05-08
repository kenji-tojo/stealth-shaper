#ifndef PRINT_SPARSE_MAT_H
#define PRINT_SPARSE_MAT_H

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <iostream>

// Inputs:
//   mat eigen sparse matrix
void print_sparse_mat(
	const Eigen::SparseMatrix<double> & mat);
#endif