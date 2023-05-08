#pragma once

#include <limits>

#include "internal/math.hpp"
#include "ray.hpp"
#include "hit.hpp"
#include "scene.hpp"
#include "trimesh.h"
#include "bsdf.hpp"
#include "light.hpp"


namespace stealth::ptrace {


template<typename Scalar>
struct Accumulator {
    Scalar value = 0.;
    Scalar clip_max = std::numeric_limits<Scalar>::max();
    Accumulator() { static_assert(std::is_floating_point_v<Scalar>); }
    Accumulator &operator+=(Scalar s) { value += math::min(clip_max, s); return *this; }
    inline void accumulate(Scalar weight, Scalar contrib) {
        contrib = contrib < clip_max ? contrib : clip_max;
        value += weight * contrib;
    }
};


template<typename Vector3, typename IRaycast>
inline bool shadow_test(
        const Vector3 &pos,
        const Vector3 &wi,
        const IRaycast &scene) {
    Ray ray{pos, wi};
    Hit hit;
    scene.raycast(ray, hit);
    return hit.dist < 1e8f;
}


template<bool shadow_test_enabled, typename Vector3>
inline typename Vector3::Scalar sample_direct_Li(
        const Hit &hit,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light) {
    if constexpr(shadow_test_enabled) {
        if (shadow_test(hit.pos, light.wi, scene)) return 0.;
    }
    return scene.bsdf(hit.mat_id).eval(hit.nrm, hit.wo, light.wi) * light.Li;
}


template<bool shadow_test_enabled, typename Vector3>
typename Vector3::Scalar sample_Li(
        Ray ray,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        unsigned int depth,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;

    Scalar Li = 0.;
    Scalar weight = 1.;
    Scalar bsdf_value, bsdf_pdf;

    for (int dd = 0; dd < depth; ++dd) {
        Hit hit;
        scene.raycast(ray, hit);

        if (!hit) break;

        auto &bsdf = scene.bsdf(hit.mat_id);

        Li += weight * sample_direct_Li<shadow_test_enabled>(hit, scene, light);

        if (dd == depth-1) break;

        ray.org = hit.pos;
        std::tie(ray.dir, bsdf_value, bsdf_pdf) = bsdf.sample_wi(hit.nrm, hit.wo, sampler);

        if (bsdf_value <= 0.) break;
        if (bsdf_pdf <= 0.) { assert(bsdf_pdf == 0.); break; }

        weight *= bsdf_value / bsdf_pdf;
    }

    return Li;
}


template<bool shadow_test_enabled, typename Vector3>
typename Vector3::Scalar sample_Lo(
        const Hit &hit,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        const unsigned int depth,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;

    if (depth == 0) return 0.;

    auto Lo = sample_direct_Li<shadow_test_enabled>(hit, scene, light);

    if (depth <= 1) return Lo;

    const auto &bsdf = scene.bsdf(hit.mat_id);
    Scalar bsdf_value, bsdf_pdf;

    Ray ray;
    ray.org = hit.pos;
    std::tie(ray.dir, bsdf_value, bsdf_pdf) = bsdf.sample_wi(hit.nrm, hit.wo, sampler);

    if (bsdf_value <= 0.) return Lo;
    if (bsdf_pdf <= 0.) { assert(bsdf_pdf == 0.); return Lo; }

    const auto weight = bsdf_value / bsdf_pdf;
    Lo += weight * sample_Li<shadow_test_enabled>(ray, scene, light, depth-1, sampler);

    return Lo;
}


template<bool shadow_test_enabled, typename Vector3>
typename Vector3::Scalar estimate_Lo(
        const TriMesh &mesh,
        int tri_id,
        const Vector3 &sensor_wo,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        unsigned int depth,
        unsigned int n_samples,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;

    auto &V = mesh.V();
    auto &F = mesh.F();
    auto &triN = mesh.triN();

    Hit hit;
    hit.nrm = triN.row(tri_id).transpose();
    hit.wo = sensor_wo;
    hit.obj_id = mesh.obj_id;
    hit.mat_id = mesh.mat_id;
    hit.prim_id = tri_id;

    if (hit.nrm.dot(hit.wo) <= 0.) return 0.;

    const Vector3 p0 = V.row(F(tri_id,0)).transpose();
    const Vector3 p1 = V.row(F(tri_id,1)).transpose();
    const Vector3 p2 = V.row(F(tri_id,2)).transpose();

    Accumulator<Scalar> Lo;
    Lo.clip_max = Scalar(1e1); // do energy clipping to alleviate fireflies

    const auto weight = Scalar(1.) / Scalar(n_samples);

    for (int ss = 0; ss < n_samples; ++ss) {
        sample_tri(p0, p1, p2, /*&out=*/hit.pos, sampler);
        hit.pos += Scalar(1e-4) * hit.nrm;

        Lo.accumulate(weight, sample_Lo<shadow_test_enabled>(hit, scene, light, depth, sampler));
    }

    return Lo.value;
}


} // namespace stealth::ptrace