#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "internal/math.hpp"
#include "internal/threads.hpp"

#include "sampler.hpp"
#include "camera.hpp"
#include "scene.hpp"
#include "trimesh.h"


namespace stealth {

class Visualizer {
public:
    using Vector3 = Eigen::Vector3f;

    int type = 0; // 0: vertex, 1: face, 2: normal
    int spp = 32;

    Vector3 back_color = {1.f, 1.f, 1.f};
    Vector3 base_color = {.5f, .5f, .5f};
    std::vector<Vector3> vertex_colors;
    std::vector<Vector3> face_colors;

    template<typename ndarray_t>
    void render(const TriMesh &mesh, const Scene<Vector3> &scene, const Camera &camera, ndarray_t &image) {
        assert(image.ndim() == 3);
        const unsigned int width = image.shape(0);
        const unsigned int height = image.shape(1);
        const unsigned int channels = image.shape(2);
        assert(channels == 3 || channels == 4);

        assert(spp%4 == 0);

        if (type == 0 && vertex_colors.size() != mesh.V().rows())
            return;

        if (type == 1 && face_colors.size() != mesh.F().rows())
            return;

        unsigned int n_threads = std::thread::hardware_concurrency();

        std::vector<Sampler<float>> sampler_pool(n_threads);

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

                const auto &V = mesh.V();
                const auto &F = mesh.F();

                if (hit.obj_id != mesh.obj_id) {
                    rgb += weight * base_color;
                }
                else if (type == 0) {
                    // vertex color
                    const int ip0 = F(hit.prim_id,0);
                    const int ip1 = F(hit.prim_id,1);
                    const int ip2 = F(hit.prim_id,2);

                    const Vector3 p0 = V.row(ip0).transpose();
                    const Vector3 p1 = V.row(ip1).transpose();
                    const Vector3 p2 = V.row(ip2).transpose();

                    const auto &q = hit.pos;

                    const auto area = (p1-p0).cross(p2-p0).norm() + 1e-5f;
                    const auto a0 = (p1-q).cross(p2-q).norm() / area;
                    const auto a1 = (p2-q).cross(p0-q).norm() / area;
                    const auto a2 = (p0-q).cross(p1-q).norm() / area;

                    const auto &vc = vertex_colors;
                    rgb += weight * (a0*vc[ip0] + a1*vc[ip1] + a2*vc[ip2]);
                }
                else if (type == 1) {
                    // face color
                    rgb += weight * face_colors[hit.prim_id];
                }
                else if (type == 2) {
                    // visualize normal
                    float lo = .25f;
                    float hi = .99f;
                    for (int ic = 0; ic < 3; ++ic) {
                        float u = math::clip(.5f*(hit.nrm[ic]+1.f), 0.f, 1.f);
                        rgb[ic] += weight * (lo+(hi-lo)*u);
                    }
                }
            }

            const float alpha = float(hit_count)/float(spp);
            if (channels == 3) {
                auto nalpha = math::max(0.,1.-alpha);
                rgb += nalpha * back_color;
                image(iw,ih,0) = rgb[0];
                image(iw,ih,1) = rgb[1];
                image(iw,ih,2) = rgb[2];
            }
            else if (channels == 4) {
                image(iw,ih,0) = alpha > 0 ? rgb[0]/alpha : 0.f;
                image(iw,ih,1) = alpha > 0 ? rgb[1]/alpha : 0.f;
                image(iw,ih,2) = alpha > 0 ? rgb[2]/alpha : 0.f;
                image(iw,ih,3) = alpha;
            }
        };

        internal::parallel_for(height, width, render_fn, n_threads);
    }

};

} // namespace stealth
