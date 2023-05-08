#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>

#include "stealth-shaper/trimesh.h"
#include "stealth-shaper/vertices_below_level.hpp"
#include "stealth-shaper/energy.hpp"
#include "stealth-shaper/direction_samler.hpp"
#include "stealth-shaper/stealth_shaper.hpp"

#include "cxxopts/cxxopts.hpp"


int main(int argc, char *argv[]) {

    using namespace Eigen;
    using namespace stealth;

    cxxopts::Options options("stealth-shaper", "Stylizing the input mesh via reflectivity optimization");
    options.add_options()
            ("obj", "The obj file for the input mesh", cxxopts::value<std::string>())
            ("s,save", "Saving intermediate shapes under ./result", cxxopts::value<bool>()->default_value("false"))
            ("v,validation", "Running validation step computing the energy value after each shape update", cxxopts::value<bool>()->default_value("false"))
            ;
    options.parse_positional({"obj"});
    auto cmd_args = options.parse(argc, argv);

    std::filesystem::create_directory("./result");


    auto mesh = std::make_shared<TriMesh>(cmd_args["obj"].as<std::string>());

    // scaling the model to reproduce our experiments
    {
        constexpr float desired_diagonal = 3.0f;

        auto V = mesh->V();
        auto F = mesh->F();

        const auto bbox_min = V.colwise().minCoeff();
        const auto bbox_max = V.colwise().maxCoeff();
        const auto diagonal = (bbox_max-bbox_min).norm();

        V *= desired_diagonal/diagonal;
        mesh = std::make_shared<TriMesh>(std::move(V), std::move(F));
    }

    mesh->print_info();

    auto bsdf = std::make_shared<PhongBSDF<TriMesh::Vector3>>();
    bsdf->albedo = 0.85f;
    bsdf->kd = 0.05f;
    bsdf->set_n(30);

    Scene<TriMesh::Vector3> scene;
    scene.add_object(mesh, bsdf);



    std::cout << "--- Optimizing reflectivity (normal-driven) ---" << std::endl;


    StealthShaperData data; // see the definition for the meaning of parameter values

    data.args.save_intermediate_shapes = cmd_args["save"].as<bool>();
    data.args.enable_validation = cmd_args["validation"].as<bool>();

    /* rapid mode */
    data.args.n_dirs = 8;
    data.args.n_dirs = 8;

    /* our paper's setting to create the 3D printed models */
    // data.args.n_dirs = 16;
    // data.args.n_dirs = 8;

    // pinned vertex indices
    data.args.normal_driven_IP = vertices_below_level(mesh->V(), mesh->avg_edge_length*0.5);
    std::cout << "Pinning down " << data.args.normal_driven_IP.size() << " vertices." << std::endl;

    ReflectivityEnergy<0/* 0: minimize, 1: maximize */> energy;
    energy.depth = 2;

    StealthDirSampler dir_sampler;
    dir_sampler.theta_zero = 20.0f;


    stealth_shaper<TriMesh::TensorNXY>(*mesh, scene, energy, dir_sampler, data);


    mesh->write_obj("./result/stealth.obj");


    if (data.args.enable_validation) {
        std::ofstream out;
        out.open("./result/energy_values.txt", std::ios::out);
        for (auto e: data.result.energy_values) { out << e << "\n"; }
    }
    if (!data.result.elapsed_secs.empty()) {
        std::ofstream out;
        out.open("./result/elapsed_secs.txt", std::ios::out);
        for (auto e: data.result.elapsed_secs) { out << e << "\n"; }
    }
    if (data.result.oldE.has_value()) {
        std::ofstream out;
        out.open("./result/subdivided_edges.txt", std::ios::out);
        const auto &oldE = data.result.oldE.value();
        for (int ii = 0; ii < oldE.rows(); ++ii) { out << oldE(ii,0) << " " << oldE(ii,1) << "\n"; }
    }
    if (data.result.newE.has_value()) {
        std::ofstream out;
        out.open("./result/inserted_edges.txt", std::ios::out);
        const auto &newE = data.result.newE.value();
        for (int ii = 0; ii < newE.rows(); ++ii) { out << newE(ii,0) << " " << newE(ii,1) << "\n"; }
    }
}