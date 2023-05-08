#include "normal_driven_precomputation.h"

void print_sparse_mat(
	const Eigen::SparseMatrix<double> & mat)
{
  using namespace Eigen;
	using namespace std;

	int numPrint = 0;
	int numToPrint = 8;
	for (int k=0; k<mat.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it)
		{
			if (numPrint < numToPrint)
				cout << "(" << it.row() << "," << it.col() << "): " << it.value() << endl;
			else if (numPrint > mat.nonZeros() - numToPrint + 1)
				cout << "(" << it.row() << "," << it.col() << "): " << it.value() << endl;
			else if (numPrint == numToPrint)
				cout << "..." << endl;
			numPrint++;
		}
	}
	cout << "matrix size: (" << mat.rows() << "," << mat.cols() << ")" << endl;
	cout << "number of nonzeros: " << mat.nonZeros() << endl;
}