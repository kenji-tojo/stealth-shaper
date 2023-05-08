#pragma once

#include <iostream>
#include <optional>

#include "internal/math.hpp"

#include "stealth_shaper_data.hpp"
#include "stealth_shaper_iteration.hpp"

#include "edge_subdivision.hpp"
#include "topology.hpp"
#include "adaptive-mesh/adaptive_mesh.h"


namespace stealth {

template<typename TensorN, int optimization_mode, typename DirectionSampler>
void stealth_shaper(
        TriMesh &mesh,
        const Scene<TriMesh::Vector3> &scene,
        const ReflectivityEnergy<optimization_mode> &energy,
        const DirectionSampler &sampler,
        StealthShaperData &data) {

    auto &args = data.args;

    auto normals = std::make_shared<TensorN>(mesh.triN_shared);

    NormalDriven normal_driven;
    normal_driven.maxIter = args.normal_driven_maxIter;
    normal_driven.lambda = args.normal_driven_lambda;

    auto denoiser = std::make_optional<L1Denoiser<float>>(mesh.V(), mesh.F());
    denoiser.value().alpha = args.denoiser_alpha;

    Eigen::VectorXf triR;


    mesh.write_obj_reference("./result/reference.obj");


    if (args.enable_coarse) {

        /* --- Coarse deformation using the vertex ARAP elements --- */

        normal_driven.type = 0; // vertex ARAP
        normal_driven.precomp(mesh.refV, mesh.F(), std::unordered_set<int>{args.normal_driven_IP.begin(), args.normal_driven_IP.end()});

        stealth_shaper_iteration(mesh, normals, scene, energy, sampler, normal_driven, denoiser.value(), data, triR);
    }

    if (args.enable_fine) {

        /* --- Fine deformation using the face ARAP elements --- */

        normal_driven.type = 1; // face ARAP
        normal_driven.precomp(mesh.refV, mesh.F());

        stealth_shaper_iteration(mesh, normals, scene, energy, sampler, normal_driven, denoiser.value(), data, triR);
    }

    if (args.enable_subdivision) {

        /* --- Adaptive edge subdivision --- */

        data.result.oldE.emplace();
        data.result.newE.emplace();
        auto &oldE = data.result.oldE.value();
        auto &newE = data.result.newE.value();

        // n_dirs and n_paths may be increased here to improve the accuracy of edge selection
        energy.estimate(mesh, scene, sampler, /*n_dirs=*/128, /*n_paths=*/16, triR.data());

        Eigen::MatrixXi uE, EF;
        Eigen::VectorXf C;
        std::vector<int> SE;
        {
            subdivided_edges(mesh.V(), mesh.F(), triR.data(), args.edge_subdivision_ratio, uE, EF, C, SE);
            auto candSE = SE;
            disjoint_edges(candSE, uE, normal_driven.IP, SE);
            std::cout << "found " << SE.size() << " disjoint edges to subdivide" << std::endl;
        }

        Eigen::MatrixXi EV;
        edge_neighbors(mesh.F(), uE, EF, EV);

        std::vector<int> insP, VMAP;
        mesh.split_edges<AdaptiveMesh>(uE, SE, insP, VMAP);

        inserted_edges(uE, EV, SE, insP, VMAP, newE);

        Eigen::VectorXi EMAP;
        Eigen::MatrixXi EI;
        igl::edge_flaps(mesh.F(), uE, EMAP, EF, EI);

        for (int &index: args.normal_driven_IP) { index = VMAP[index]; }

        edge_set_difference(uE, newE, oldE);



        /* --- Update the normal tensor and denoiser for the subdivided mesh --- */
        normals = std::make_shared<TensorN>(mesh.triN_shared);

        denoiser.reset();
        denoiser.emplace(mesh.V(), mesh.F());
        denoiser.value().alpha = args.denoiser_alpha;

        normal_driven.type = 1; // face ARAP
        normal_driven.precomp(mesh.refV, mesh.F(), std::unordered_set<int>{args.normal_driven_IP.begin(), args.normal_driven_IP.end()});

        mesh.write_obj_reference("./result/reference_subdivided.obj");

        stealth_shaper_iteration(mesh, normals, scene, energy, sampler, normal_driven, denoiser.value(), data, triR);
    }


    if (args.save_intermediate_shapes) {
        std::string path = "./result/";
        path += std::to_string(data.result.iter_id);
        path += ".obj";
        std::cout << "writing to " << path << std::endl;
        mesh.write_obj(path);
    }

    if (args.enable_validation) {
        energy.estimate(mesh, scene, sampler, args.n_dirs_validation, args.n_paths_validation, triR.data());
        data.result.energy_values.push_back(triR.sum());
        assert(data.result.energy_values.size() == args.n_iter*3+1);
    }

    assert(data.result.elapsed_secs.size() == args.n_iter*3+1);
}

} // namespace stealth