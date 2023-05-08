#include "normal_driven_precomputation.h"

void normal_driven_precomputation(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const int & type,
    normal_driven_data & data)
{
    if (type == 0)
        normal_driven_precomputation_vertex(V,F,data);
    else if (type == 1)
        normal_driven_precomputation_face(V,F,data);
}

void normal_driven_precomputation_vertex(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    normal_driven_data & data)
{
    using namespace Eigen;
	using namespace std;
    // cout << "use vertex arap\n"; 
    int nV = V.rows();
    data.reset();

    igl::per_vertex_normals(V,F, data.N);

    MatrixXd cotanW;
    igl::cotmatrix_entries(V,F,cotanW);

    SparseMatrix<double> M;
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_BARYCENTRIC,M);
    data.A = M.diagonal();
    data.totalArea = data.A.sum();

    vector<vector<int>> adjFList, VI;
    igl::vertex_triangle_adjacency(V.rows(),F,adjFList,VI);

    // initialize everything 
    data.E_N.resize(V.rows());
    data.Wvec.resize(V.rows());
    data.dV.resize(V.rows());

    vector<Triplet<double>> QIJV, KIJV;
    QIJV.reserve(V.rows()*4*3*4); // reserve enough size (4 items per edge of f_k)
    KIJV.reserve(V.rows()*18*3*4); // reserve enough size (18 items per edge of f_k)

    vector<int> adjFk;
    for (int kk=0; kk<V.rows();  kk++)
    {
        adjFk = adjFList[kk];

        // get edge vectors, cotan weights, area for N_k 
        data.E_N[kk].resize(adjFk.size()*3, 2);
        data.Wvec[kk].resize(adjFk.size()*3);
        for (int jj=0; jj<adjFk.size(); jj++)
        {
            int v0 = F(adjFk[jj],0);
            int v1 = F(adjFk[jj],1);
            int v2 = F(adjFk[jj],2);

            // compute adjacent half-edge indices of a vertex
            data.E_N[kk](3*jj  ,0) = v0;
            data.E_N[kk](3*jj  ,1) = v1;
            data.E_N[kk](3*jj+1,0) = v1;
            data.E_N[kk](3*jj+1,1) = v2;
            data.E_N[kk](3*jj+2,0) = v2;
            data.E_N[kk](3*jj+2,1) = v0;

            // compute WVec = vec(W)
            data.Wvec[kk](3*jj  ) = cotanW(adjFk[jj],2);
            data.Wvec[kk](3*jj+1) = cotanW(adjFk[jj],0);
            data.Wvec[kk](3*jj+2) = cotanW(adjFk[jj],1);
        }

        // compute dV
        data.dV[kk].resize(3, adjFk.size()*3);
        {
            MatrixXd E0_V, E1_V;
            igl::slice(V,data.E_N[kk].col(0),1,E0_V);
            igl::slice(V,data.E_N[kk].col(1),1,E1_V);
            data.dV[kk] = (E1_V - E0_V).transpose();
        }

        // precomputation for the global step (Q and K)
        {
            MatrixXi E = data.E_N[kk];
            VectorXd Wvec = data.Wvec[kk];
            int nE = E.rows(); // nE = 3

            for (int ii=0; ii<nE; ii++)
            {
                // Q
                QIJV.push_back(Triplet<double>(E(ii,0), E(ii,1), Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,1), E(ii,0), Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,0), E(ii,0), -Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,1), E(ii,1), -Wvec(ii)));

                //K
                for (int dim=0; dim<3; dim++)
                {
                    double val0 = (Wvec(ii)*V(E(ii,0),dim) - Wvec(ii)*V(E(ii,1),dim));
                    double val1 = (Wvec(ii)*V(E(ii,1),dim) - Wvec(ii)*V(E(ii,0),dim));
                    KIJV.push_back(Triplet<double>(dim+9*kk,   E(ii,0), val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk,   E(ii,1), val1));
                    KIJV.push_back(Triplet<double>(dim+9*kk+3, E(ii,0)+nV, val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk+3, E(ii,1)+nV, val1));
                    KIJV.push_back(Triplet<double>(dim+9*kk+6, E(ii,0)+2*nV, val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk+6, E(ii,1)+2*nV, val1));
                }
            }
        }
    }

    // construct LHS for global step
    data.Q.resize(V.rows(), V.rows());
    data.Q.setFromTriplets(QIJV.begin(),QIJV.end());
    
    // construct K1 K2 for the RHS
    data.K.resize(9*nV, 3*nV);
    data.K.setFromTriplets(KIJV.begin(),KIJV.end());

    igl::min_quad_with_fixed_precompute(data.Q,data.b,SparseMatrix<double>(),false,data.solver_data);
}

void normal_driven_precomputation_face(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    normal_driven_data & data)
{
    using namespace Eigen;
	using namespace std;
    // cout << "use face arap\n"; 
    int nF = F.rows();
    int nV = V.rows();
    data.reset();

    igl::doublearea(V,F,data.A); 
    data.A = data.A / 2; // face areas
    data.totalArea = data.A.sum();

    igl::triangle_triangle_adjacency(F,data.AF);

    // compute some mesh info
    igl::per_face_normals(V,F, data.N); // original face normals

    MatrixXd cotanW;
    igl::cotmatrix_entries(V,F,cotanW);

    // initialize everything 
    data.E_N.resize(F.rows());
    data.Wvec.resize(F.rows());
    data.dV.resize(F.rows());

    vector<Triplet<double>> QIJV, KIJV;
    QIJV.reserve(F.rows()*4*3); // reserve enough size (4 items per edge of f_k)
    KIJV.reserve(F.rows()*18*3); // reserve enough size (18 items per edge of f_k)

    SparseMatrix<double> VT_sparse = V.transpose().sparseView(); // transpose V (for other computation)
    for (int kk=0; kk<nF; kk++)
    {
        // vertices of f_k
        int v0 = F(kk, 0);
        int v1 = F(kk, 1);
        int v2 = F(kk, 2);

        VectorXi adjFk;
        adjFk.resize(1);
        adjFk(0) = kk;

        // get edge vectors, cotan weights, area for N_k 
        data.E_N[kk].resize(adjFk.size()*3, 2);
        data.Wvec[kk].resize(adjFk.size()*3);
        for (int jj=0; jj<adjFk.size(); jj++)
        {
            int v0 = F(adjFk[jj],0);
            int v1 = F(adjFk[jj],1);
            int v2 = F(adjFk[jj],2);

            // compute adjacent half-edge indices of a vertex
            data.E_N[kk](3*jj  ,0) = v0;
            data.E_N[kk](3*jj  ,1) = v1;
            data.E_N[kk](3*jj+1,0) = v1;
            data.E_N[kk](3*jj+1,1) = v2;
            data.E_N[kk](3*jj+2,0) = v2;
            data.E_N[kk](3*jj+2,1) = v0;

            // compute WVec = vec(W)
            data.Wvec[kk](3*jj  ) = cotanW(adjFk[jj],2);
            data.Wvec[kk](3*jj+1) = cotanW(adjFk[jj],0);
            data.Wvec[kk](3*jj+2) = cotanW(adjFk[jj],1);
        }

        // compute dV
        data.dV[kk].resize(3, adjFk.size()*3);
        {
            MatrixXd E0_V, E1_V;
            igl::slice(V,data.E_N[kk].col(0),1,E0_V);
            igl::slice(V,data.E_N[kk].col(1),1,E1_V);
            data.dV[kk] = (E1_V - E0_V).transpose();
        }

        // precomputation for the global step (Q and K)
        {
            MatrixXi E = data.E_N[kk];
            VectorXd Wvec = data.Wvec[kk];
            int nE = E.rows(); // nE = 3

            for (int ii=0; ii<nE; ii++)
            {
                // Q
                QIJV.push_back(Triplet<double>(E(ii,0), E(ii,1), Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,1), E(ii,0), Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,0), E(ii,0), -Wvec(ii)));
                QIJV.push_back(Triplet<double>(E(ii,1), E(ii,1), -Wvec(ii)));

                //K
                for (int dim=0; dim<3; dim++)
                {
                    double val0 = (Wvec(ii)*V(E(ii,0),dim) - Wvec(ii)*V(E(ii,1),dim));
                    double val1 = (Wvec(ii)*V(E(ii,1),dim) - Wvec(ii)*V(E(ii,0),dim));
                    KIJV.push_back(Triplet<double>(dim+9*kk,   E(ii,0), val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk,   E(ii,1), val1));
                    KIJV.push_back(Triplet<double>(dim+9*kk+3, E(ii,0)+nV, val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk+3, E(ii,1)+nV, val1));
                    KIJV.push_back(Triplet<double>(dim+9*kk+6, E(ii,0)+2*nV, val0));
                    KIJV.push_back(Triplet<double>(dim+9*kk+6, E(ii,1)+2*nV, val1));
                }
            }
        }
    }

    // construct LHS for global step
    data.Q.resize(V.rows(), V.rows());
    data.Q.setFromTriplets(QIJV.begin(),QIJV.end());
    
    // construct K1 K2 for the RHS
    data.K.resize(9*nF, 3*nV);
    data.K.setFromTriplets(KIJV.begin(),KIJV.end());

    // precomputation
    igl::min_quad_with_fixed_precompute(data.Q,data.b,SparseMatrix<double>(),false,data.solver_data);
}