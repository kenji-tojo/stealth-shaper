#pragma once

#include <iostream>
#include <vector>

#include "internal/math.hpp"
#include "internal/threads.hpp"
#include "internal/timer.hpp"

#include "bsdf.hpp"
#include "trimesh.h"
#include "sampler.hpp"
#include "ptrace.hpp"
#include "backprop.hpp"
#include "direction_samler.hpp"


namespace stealth {


template<int optimization_mode = 0 /* 0: deflection, 1: reflection */>
class ReflectivityEnergy {
public:
    using Vector3 = TriMesh::Vector3;
    using Scalar = typename Vector3::Scalar;

    /*
     * SCALING: A scalar multiplied to the final energy value,
     * which corresponds to the total surface area of the input mesh.
     * Although the scaling does not affect the mathematical formulation,
     * it will help roughly reproduce the energy values reported in the paper.
     */
    constexpr static float SCALING = 5.0f;

    unsigned int depth = 2; // the depth of paths in reflective light-transport simulation

    unsigned int n_threads = std::thread::hardware_concurrency();

    ReflectivityEnergy() {
        using namespace std;
        static_assert(optimization_mode == 0 || optimization_mode == 1);
        cout << "ReflectivityEnergy: mode = " << optimization_mode;
        if constexpr(optimization_mode == 0) { cout << " (deflection)"; } else { cout << " (reflection)"; }
        cout << endl;
    }

    template<typename DirectionSampler>
    void estimate(
            const TriMesh &mesh,
            const Scene<Vector3> &scene,
            const DirectionSampler &dir_sampler,
            unsigned int n_dirs,
            unsigned int n_paths,
            Scalar *triR /* #F list of energy values on each triangle */,
            int random_seed = -1) const {

        std::vector<Sampler<Scalar>> sampler_pool(n_threads);
        if (random_seed >= 0) {
            for (auto &s: sampler_pool) s.set_seed(random_seed);
        }

        std::vector<std::vector<Vector3>> sensor_wo_pool(n_threads);
        std::vector<std::vector<Vector3>> light_wi_pool(n_threads);

        std::fill(triR, triR+mesh.F().rows(), 0.);

        auto kernel = [&] (int tri_id, int tid) {
            auto &sampler = sampler_pool[tid];
            auto &sensor_wo = sensor_wo_pool[tid];
            auto &light_wi = light_wi_pool[tid];

            if constexpr(std::is_same_v<DirectionSampler, StealthDirSampler>) {
                dir_sampler.sample(n_dirs, sensor_wo, light_wi, sampler);
            }
            else {
                static_assert(std::is_same_v<DirectionSampler, SunlightDirSampler>);
                const Vector3 p0 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 p1 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 p2 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 c = Scalar(1./3.) * (p0 + p1 + p2);
                dir_sampler.sample(n_dirs, c, sensor_wo, light_wi, sampler);
            }

            Light<Vector3> light;
            light.Li = 1.;

//            const auto weight = Scalar(1.) / Scalar(n_dirs); // this would be more natural mathematically
            const auto weight = Scalar(SCALING) / Scalar(n_dirs); // see the explanation above about SCALING

            for (int ii = 0; ii < n_dirs; ++ii) {
                light.wi = light_wi[ii];

                constexpr bool shadow_test_enabled = true;
                const auto Lo = ptrace::estimate_Lo<shadow_test_enabled>(mesh, tri_id, sensor_wo[ii], scene, light, depth, n_paths, sampler);

                triR[tri_id] += weight * Scalar(.5) * mesh.A[tri_id] * Lo * Lo;
            }
        };

        using namespace std;

        cout << "estimating ";
        if constexpr(optimization_mode == 0) { cout << "deflection"; } else { cout << "reflection"; }
        cout << " energy: "
             << "number of faces = " << mesh.F().rows() << ", "
             << "n_dirs = " << n_dirs << ", "
             << "n_paths = " << n_paths << endl;

        internal::parallel_for(mesh.F().rows(), kernel, n_threads);
    }


    template<typename GradientAccumulator, typename DirectionSampler, typename Tensor>
    void estimate_and_backprop(
            const TriMesh &mesh,
            const std::shared_ptr<Tensor> &parameters,
            const Scene<Vector3> &scene,
            const DirectionSampler &dir_sampler,
            unsigned int n_dirs,
            unsigned int n_paths,
            Scalar *triR /* #F list of energy values on each triangle */,
            int random_seed = -1) const {

        std::vector<Sampler<Scalar>> sampler_pool(n_threads);
        if (random_seed >= 0) {
            for (auto &s: sampler_pool) s.set_seed(random_seed);
        }

        std::vector<std::vector<Vector3>> sensor_wo_pool(n_threads);
        std::vector<std::vector<Vector3>> light_wi_pool(n_threads);

        std::vector<GradientAccumulator> GA_pool;
        for (int ii = 0; ii < n_threads; ++ii) {
            if constexpr(std::is_same_v<GradientAccumulator, TriMesh::GradAccV>) {
                GA_pool.emplace_back(parameters, mesh.F_shared, mesh.triN_shared, mesh.obj_id);
            }
            else {
                static_assert(std::is_same_v<GradientAccumulator, TriMesh::GradAccN>);
                GA_pool.emplace_back(parameters, mesh.obj_id);
            }
        }

        std::fill(triR, triR+mesh.F().rows(), 0.);

        auto kernel = [&] (int tri_id, int tid) {
            auto &sampler = sampler_pool[tid];
            auto &sensor_wo = sensor_wo_pool[tid];
            auto &light_wi = light_wi_pool[tid];
            auto &GA = GA_pool[tid];

            if constexpr(std::is_same_v<DirectionSampler, StealthDirSampler>) {
                dir_sampler.sample(n_dirs, sensor_wo, light_wi, sampler);
            }
            else {
                static_assert(std::is_same_v<DirectionSampler, SunlightDirSampler>);
                const Vector3 p0 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 p1 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 p2 = mesh.V().row(mesh.F()(tri_id,0));
                const Vector3 c = Scalar(1./3.) * (p0 + p1 + p2);
                dir_sampler.sample(n_dirs, c, sensor_wo, light_wi, sampler);
            }

            Light<Vector3> light;
            light.Li = 1.;

//            const auto weight = Scalar(1.) / Scalar(n_dirs); // this would be more natural mathematically
            const auto weight = Scalar(SCALING) / Scalar(n_dirs); // see the explanation above about SCALING

            for (int ii = 0; ii < n_dirs; ++ii) {
                light.wi = light_wi[ii];

                constexpr bool shadow_test_enabled = true;
                const auto Lo = ptrace::estimate_Lo<shadow_test_enabled>(mesh, tri_id, sensor_wo[ii], scene, light, depth, n_paths, sampler);

                triR[tri_id] += weight * Scalar(.5) * mesh.A[tri_id] * Lo * Lo;

                Scalar Ae;
                if constexpr(optimization_mode == 0) {
                    // E = A * Lo * Lo
                    Ae = weight * mesh.A[tri_id] * Lo;
                }
                else {
                    // E = A / (Lo + eps)
                    const auto eps = Scalar(1e-5);
                    Ae = weight * mesh.A[tri_id] * Scalar(-1.) / (Lo + eps) / (Lo + eps);
                }
                backprop::propagate_adjoint_Lo<shadow_test_enabled>(Ae, mesh, tri_id, sensor_wo[ii], scene, light, GA, depth, n_paths, sampler);
            }
        };

        using namespace std;

        cout << "estimating and back-propagating ";
        if constexpr(optimization_mode == 0) { cout << "deflection"; } else { cout << "reflection"; }
        cout << " energy: "
             << "number of faces = " << mesh.F().rows() << ", "
             << "n_dirs = " << n_dirs << ", "
             << "n_paths = " << n_paths << endl;

        internal::parallel_for(mesh.F().rows(), kernel, n_threads);
        for (auto &GA: GA_pool) GA.flush();
    }

};


} // namespace stealth