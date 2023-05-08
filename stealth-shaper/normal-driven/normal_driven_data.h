#ifndef NORMAL_DRIVEN_DATA_H
#define NORMAL_DRIVEN_DATA_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <limits>
#include <igl/min_quad_with_fixed.h>

struct normal_driven_data
{
	// user should tune these
	double lambda = 1;
	double developable_threshold = 0.0;
	double objVal = 0;
	double reldV = std::numeric_limits<float>::max();
	double totalArea = 0;
	bool computeEnergy = false;
	const bool useScale = false; /* [ Stealth Shaper ]: We tuned off the ACAP deformation to make the dependency minimal. */

	std::vector<double> objHis;
	std::vector<Eigen::MatrixXi> E_N;
	std::vector<Eigen::VectorXd> Wvec;
	std::vector<Eigen::MatrixXd> dV, UHis;

	// Eigen::SparseMatrix<double> K1, K2, Q1, Q2;
	Eigen::SparseMatrix<double> K, Q;
	Eigen::MatrixXd N;
	Eigen::VectorXd objValVec, A;

	// bool useBc = false;
	Eigen::MatrixXd bc;
	Eigen::VectorXi b;
	Eigen::MatrixXi AF;

	igl::min_quad_with_fixed_data<double> solver_data;

	void reset()
	{
		// usually these don't need to tune
		objVal = 0;
		totalArea = 0;
		double reldV = std::numeric_limits<float>::max();

		objHis.clear();
		E_N.clear(); 
		dV.clear();
		UHis.clear();
		Wvec.clear();
		// LHS = Eigen::SparseMatrix<double>();
		Q = Eigen::SparseMatrix<double>();
		K = Eigen::SparseMatrix<double>();
		N = Eigen::MatrixXd();
		// gaussPos = Eigen::MatrixXd();
		AF = Eigen::MatrixXi();
		A = Eigen::VectorXd();
		objValVec = Eigen::VectorXd();

		// igl::min_quad_with_fixed_data<double> solver_data;
		// solver_data = igl::min_quad_with_fixed_data<double>();
	}
};

#endif