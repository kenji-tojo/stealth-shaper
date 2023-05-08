#pragma once

#include "ptrace.hpp"
#include "tensor.hpp"
#include "scene.hpp"


namespace stealth::backprop {


template<bool shadow_test_enabled, typename GradientAccumulator, typename Vector3>
inline void adjoint_direct_Li(
        typename Vector3::Scalar weight,
        const Hit &hit,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        GradientAccumulator &GA) {
    if (GA.obj_id != hit.obj_id) return;
    if constexpr(shadow_test_enabled) {
        if (ptrace::shadow_test(hit.pos, light.wi, scene)) return;
    }
    GA.accumulate(hit.prim_id, weight, light.Li * scene.bsdf(hit.mat_id).dFdN(hit.nrm, hit.wo, light.wi));
}


template<bool shadow_test_enabled, typename GradientAccumulator, typename Vector3>
void adjoint_Li(
        typename Vector3::Scalar weight,
        Ray ray,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        GradientAccumulator &GA,
        unsigned int depth,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;

    Scalar bsdf_value, bsdf_pdf;

    for (int dd = 0; dd < depth; ++dd) {
        Hit hit;
        scene.raycast(ray, hit);

        if (!hit) break;

        auto &bsdf = scene.bsdf(hit.mat_id);

        adjoint_direct_Li<shadow_test_enabled>(weight, hit, scene, light, GA);

        if (dd == depth-1) break;

        ray.org = hit.pos;
        std::tie(ray.dir, bsdf_value, bsdf_pdf) = bsdf.sample_wi(hit.nrm, hit.wo, sampler);

        if (bsdf_value <= 0.) break;
        if (bsdf_pdf <= 0.) { assert(bsdf_pdf == 0.); break; }

        if (GA.obj_id == hit.obj_id) {
            const auto Li = ptrace::sample_Li<shadow_test_enabled>(ray, scene, light, /*depth=*/depth-dd-1, sampler);
            GA.accumulate(hit.prim_id, weight / bsdf_pdf, Li * bsdf.dFdN(hit.nrm, hit.wo, ray.dir));
        }

        weight *= bsdf_value / bsdf_pdf;
    }
}


template<bool shadow_test_enabled, typename GradientAccumulator, typename Vector3>
void adjoint_Lo(
        typename Vector3::Scalar weight,
        const Hit &hit,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        GradientAccumulator &GA,
        const unsigned int depth,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;

    if (depth == 0) return;

    adjoint_direct_Li<shadow_test_enabled>(weight, hit, scene, light, GA);

    if (depth <= 1) return;

    const auto &bsdf = scene.bsdf(hit.mat_id);
    Scalar bsdf_value, bsdf_pdf;

    Ray ray;
    ray.org = hit.pos;
    std::tie(ray.dir, bsdf_value, bsdf_pdf) = bsdf.sample_wi(hit.nrm, hit.wo, sampler);

    if (bsdf_value <= 0.) return;
    if (bsdf_pdf <= 0.) { assert(bsdf_pdf == 0.); return; }

    if (GA.obj_id == hit.obj_id) {
        const auto Li = ptrace::sample_Li<shadow_test_enabled>(ray, scene, light, depth-1, sampler);
        GA.accumulate(hit.prim_id, weight / bsdf_pdf, Li * bsdf.dFdN(hit.nrm, hit.wo, ray.dir));
    }

    weight *= bsdf_value / bsdf_pdf;
    adjoint_Li<shadow_test_enabled>(weight, ray, scene, light, GA, depth-1, sampler);
}


template<bool shadow_test_enabled, typename GradientAccumulator, typename Vector3>
void propagate_adjoint_Lo(
        const typename Vector3::Scalar Ae,
        const TriMesh &mesh,
        int tri_id,
        const Vector3 &sensor_wo,
        const Scene<Vector3> &scene,
        const Light<Vector3> &light,
        GradientAccumulator &GA,
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

    if (hit.nrm.dot(hit.wo) <= 0.) return;

    const Vector3 p0 = V.row(F(tri_id,0)).transpose();
    const Vector3 p1 = V.row(F(tri_id,1)).transpose();
    const Vector3 p2 = V.row(F(tri_id,2)).transpose();

    const auto weight = Ae / Scalar(n_samples);

    for (int ss = 0; ss < n_samples; ++ss) {
        sample_tri(p0, p1, p2, /*&out=*/hit.pos, sampler);
        hit.pos += Scalar(1e-4) * hit.nrm;

        adjoint_Lo<shadow_test_enabled>(weight, hit, scene, light, GA, depth, sampler);
    }
}


} // namespace stealth::backprop