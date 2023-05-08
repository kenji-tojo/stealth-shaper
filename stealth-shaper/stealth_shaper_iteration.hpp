#pragma once

#include <vector>
#include <string>
#include <memory>

#include "internal/math.hpp"

#include "trimesh.h"
#include "scene.hpp"
#include "energy.hpp"
#include "direction_samler.hpp"
#include "denoiser.hpp"
#include "squared_normal_difference.hpp"
#include "stealth_shaper_data.hpp"
#include "normal-driven/normal_driven.hpp"


namespace stealth {


template<int optimization_type, typename DirectionSampler, typename DenoiserScalar, typename DerivedR>
void stealth_shaper_iteration(
        TriMesh &mesh,
        const std::shared_ptr<TriMesh::TensorNXY> &normals,
        const Scene<TriMesh::Vector3> &scene,
        const ReflectivityEnergy<optimization_type> &energy,
        const DirectionSampler &sampler,
        NormalDriven &normal_driven,
        L1Denoiser<DenoiserScalar> &denoiser,
        StealthShaperData &data,
        Eigen::PlainObjectBase<DerivedR> &triR) {

    using namespace Eigen;

    const auto &args = data.args;

    triR.resize(mesh.F().rows());

    TriMesh::MatrixN tarN;
    MatrixXd defV;

    const int total_iter = args.n_iter * (int(args.enable_coarse) + int(args.enable_fine) + int(args.enable_subdivision));
    const int start_iter = data.result.iter_id;

    internal::Timer timer;

    for (int iter = 0; iter < args.n_iter; ++iter) {
        std::cout << "-- iteration #" << start_iter+iter+1 << "/" << total_iter << " --" << std::endl;

        if (args.save_intermediate_shapes) {
            std::string path = "./result/";
            path += std::to_string(start_iter+iter);
            path += ".obj";
            std::cout << "writing to " << path << std::endl;
            mesh.write_obj(path);
        }

        if (args.enable_validation) {
            energy.estimate(mesh, scene, sampler, args.n_dirs_validation, args.n_paths_validation, triR.data());
            data.result.energy_values.push_back(triR.sum());
            std::cout << "validation: "
                      << "energy = " << data.result.energy_values.back()
                      << std::endl;
        }

        timer.restart();

        for (int igrd = 0; igrd < args.n_grad; ++igrd) {

            energy.template estimate_and_backprop<TriMesh::GradAccN>(mesh, normals, scene, sampler, args.n_dirs, args.n_paths, triR.data());

            backprop_squared_normal_difference(args.beta, mesh, normals);

            normals->descent(args.step_size);
        }

        denoiser.denoise(*normals->mat, tarN, /*n_iter=*/40);

        defV = mesh.V().cast<double>();
        normal_driven.update_vertices(mesh.refV, mesh.F(), tarN.cast<double>(), defV);

        mesh.V() = defV.cast<float>();
        mesh.update_positions();
        normals->reset_frames();

        // update the denoiser data for the modified vertex positions
        denoiser.update_positions(mesh.V(), mesh.F());

        timer.stop();

        auto curr_sec = data.result.elapsed_secs.back();
        data.result.elapsed_secs.push_back(curr_sec+timer.elapsed_sec());
    }

    data.result.iter_id += args.n_iter;
}


} // namespace stealth