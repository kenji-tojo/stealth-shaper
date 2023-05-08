#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "internal/math.hpp"
#include "internal/threads.hpp"

#include "sampler.hpp"
#include "camera.hpp"
#include "scene.hpp"
#include "backprop.hpp"


namespace stealth {

class Renderer {
public:
    using Vector3 = Eigen::Vector3f;

    unsigned int spp = 8;
    unsigned int depth = 1;

    Light<Vector3> light;

    float clip_max = 1e1;

    template<typename ndarray_t>
    void render(
            const Scene<Vector3> &scene,
            const Camera &camera,
            ndarray_t &image,
            int random_seed = -1) {
        assert(image.ndim() == 3);
        const unsigned int width = image.shape(0);
        const unsigned int height = image.shape(1);
        const unsigned int channels = image.shape(2);
        assert(channels==1 || channels==3 || channels==4);

        assert(spp%4 == 0);

        unsigned int n_threads = std::thread::hardware_concurrency();
        if (random_seed >= 0) { n_threads = 1; }

        std::vector<Sampler<float>> sampler_pool(n_threads);
        if (random_seed >= 0) {
            std::cout << "Renderer::render [ Warning ]: setting random seed to " << random_seed << std::endl;
            for (auto &s: sampler_pool) { s.set_seed(random_seed); }
        }

        const float weight = 1.f/float(spp);

        auto render_fn = [&] (int iw, int ih, int tid) {
            Vector3 rgb = {0.f, 0.f, 0.f};

            unsigned int hit_count = 0;

            auto &sampler = sampler_pool[tid];

            for (int ss = 0; ss < spp; ++ss) {
                // stratified sampling
                const auto x = -1.f+2.f*(float(iw*2)+float((ss/1)%2)+sampler.sample())/float(width*2);
                const auto y = -1.f+2.f*(float(ih*2)+float((ss/2)%2)+sampler.sample())/float(height*2);

                Ray ray = camera.spawn_ray(y,-x);
                Hit hit;
                scene.raycast(ray, hit);

                if (!hit) continue;

                hit_count += 1;

                auto Lo = ptrace::sample_Lo<true>(hit, scene, light, depth, sampler);
                Lo = Lo < clip_max ? Lo : clip_max;
                rgb += weight * Vector3::Ones() * Lo;
            }

            if (channels == 1) {
                image(iw,ih,0) = rgb.sum()/3.f;
            }
            else if (channels == 3) {
                image(iw,ih,0) = rgb[0];
                image(iw,ih,1) = rgb[1];
                image(iw,ih,2) = rgb[2];
            }
            else if (channels == 4) {
                const float alpha = float(hit_count)/float(spp);
                image(iw,ih,0) = alpha > 0 ? rgb[0]/alpha : 0.f;
                image(iw,ih,1) = alpha > 0 ? rgb[1]/alpha : 0.f;
                image(iw,ih,2) = alpha > 0 ? rgb[2]/alpha : 0.f;
                image(iw,ih,3) = alpha;
            }
        };

        internal::parallel_for(height, width, render_fn, n_threads);
    }

    template<typename ndarray_t, typename GradHashMap>
    void adjoint(
            ndarray_t &image_adj,
            const Scene<Vector3> &scene,
            const Camera &camera,
            std::vector<GradHashMap> &nrm_adj_pool,
            int random_seed = -1) {

        assert(image_adj.ndim() == 2);
        const unsigned int width  = image_adj.shape(0);
        const unsigned int height = image_adj.shape(1);

        const unsigned int n_threads = nrm_adj_pool.size();
        std::vector<Sampler<float>> sampler_pool(n_threads);
        if (random_seed >= 0) {
            std::cout << "Renderer::adjoint [ Warning ]: setting random seed to " << random_seed << std::endl;
            for (auto &s: sampler_pool) { s.set_seed(random_seed); }
        }

        auto adjoint_fn = [&] (int iw, int ih, int tid) {
            auto &sampler = sampler_pool[tid];
            auto nrm_adj  = nrm_adj_pool[tid];

            const float weight = image_adj(iw,ih) / float(spp);

            for (int ss = 0; ss < spp; ++ss) {
                // stratified sampling
                const auto x = -1.f+2.f*(float(iw*2)+float((ss/1)%2)+sampler.sample())/float(width*2);
                const auto y = -1.f+2.f*(float(ih*2)+float((ss/2)%2)+sampler.sample())/float(height*2);

                Ray ray = camera.spawn_ray(y,-x);
                Hit hit;
                scene.raycast(ray, hit);

                if (!hit) continue;

                constexpr bool shadow_test_enabled = true;
                backprop::adjoint_Lo<shadow_test_enabled>(weight, hit, scene, light, nrm_adj, depth, sampler);
            }
        };

        internal::parallel_for(height, width, adjoint_fn, n_threads);
        for (auto &nrm_adj: nrm_adj_pool) nrm_adj.flush();
    }

};

} // namespace stealth