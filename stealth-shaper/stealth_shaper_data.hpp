#pragma once

#include <vector>
#include <optional>

#include <Eigen/Dense>


namespace stealth {

class StealthShaperData {
public:

    struct Args {
        // arguments that control the optimization

        unsigned int n_iter = 30; // # vertex updates by minimizing the ARAP-based energy
        unsigned int n_grad = 8; // # normal updates before each vertex update

        unsigned int n_dirs = 16; // # incoming light directions sampled for a single reflectivity derivative estimation
        unsigned int n_paths = 8; // # paths sampled for each incoming direction

        bool enable_validation = false; // whether to compute energy values during optimization for validation
        unsigned int n_dirs_validation = 16; // # samples for validation
        unsigned int n_paths_validation = 8; // # samples for validation

        float beta = 0.1f; // strength of normal difference regularization of target normals

        float step_size = 1e2f; // step size for the gradient descent of target normals

        float denoiser_alpha = 250.0f; // strength of L1 regularization for the normal denoiser

        float edge_subdivision_ratio = 0.05f; // ratio of edges to be processed in the adaptive subdivision step

        // the following three are passed to the code of [Liu et al. 2022] under normal-driven/
        int normal_driven_maxIter = 20; // number of iteration for each normal-driven vertex update.
        double normal_driven_lambda = 1e3; // strength of normal alignment
        std::vector<int> normal_driven_IP; // vertex indices to be fixed during the deformation (IP is for "Indices to be Pinned down")

        bool save_intermediate_shapes = false; // the intermediate shapes will be saved in an OBJ file
        bool enable_coarse = true;
        bool enable_fine = true;
        bool enable_subdivision = true;

    } args;

    struct Results {
        // intermediate results of the optimization
        // maybe useful for analyzing the algorithm

        int iter_id = 0; // current iteration count

        std::vector<double> energy_values; // history of energy values at each shape update
        std::vector<double> elapsed_secs = {0.0}; // history of elapsed seconds at each shape update

        std::optional<Eigen::MatrixXi> oldE; // #E by 2 list representing edges of the INITIAL mesh that were subdivided
        std::optional<Eigen::MatrixXi> newE; // #E by 2 list representing edges of the SUBDIVIDED mesh that were newly introduced

        void clear () {
            iter_id = 0;
            energy_values.clear();
            energy_values.shrink_to_fit();
            elapsed_secs = {0.0};
            oldE.reset();
            newE.reset();
        }

    } result;

};

} // stealth