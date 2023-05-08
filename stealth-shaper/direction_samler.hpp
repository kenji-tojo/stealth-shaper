#pragma once

#include <iostream>
#include <vector>

#include "internal/math.hpp"


namespace stealth {


class StealthDirSampler {
public:
    float theta_zero = 20.f;

    template<typename Vector3>
    void sample(
            unsigned int n_dirs,
            std::vector<Vector3> &sensor_wo,
            std::vector<Vector3> &light_wi,
            Sampler<typename Vector3::Scalar> &sampler) const {

        namespace math = internal::math;
        using Scalar = typename Vector3::Scalar;

        n_dirs = 2*(n_dirs/2);

        sensor_wo.resize(n_dirs);
        light_wi.resize(n_dirs);

        for (int ii = 0; ii < n_dirs; ++ii) {
            auto &wo = sensor_wo[ii];
            auto &wi = light_wi[ii];

            Scalar phi = (Scalar(ii/2) + sampler.sample()) / Scalar(n_dirs/2);
            phi *= math::Pi2<Scalar>;

            Scalar z = -1. + (Scalar(ii%2) + sampler.sample());
            z *= std::sin(math::Pi<Scalar> * theta_zero / Scalar(180.));

            Scalar rxy = std::sqrt(math::max(0., 1.-z*z));

            wo[0] = wi[0] = rxy * std::cos(phi);
            wo[1] = wi[1] = rxy * std::sin(phi);
            wo[2] = wi[2] = z;
        }
    }

};


class SunlightDirSampler {
public:
    int type = 0; // 0: point, 1: line

    Eigen::Vector3f east = decltype(east)::UnitX();
    Eigen::Vector3f noon = Eigen::Vector3f{0.,-1.,1.}.normalized();

    Eigen::Vector3f target = decltype(target)::Zero();

    template<typename Vector3>
    void sample(
            unsigned int n_dirs,
            const Vector3 &reflection_pos,
            std::vector<Vector3> &sensor_wo,
            std::vector<Vector3> &light_wi,
            Sampler<typename Vector3::Scalar> &sampler) const {

        using namespace std;

        sunlight_dirs(n_dirs, light_wi, sampler);

        if (type == 0) {
            direct_to_point(n_dirs, reflection_pos, sensor_wo, sampler);
        }
        else if (type == 1) {
            direct_to_line(n_dirs, reflection_pos, sensor_wo, sampler);
        }
        else {
            cerr << "SunlightDirSampler::sample [ Warning ]: unknown sampling type " << type << " is specified. falling back to type = 0" << endl;
            direct_to_point(n_dirs, reflection_pos, sensor_wo, sampler);
        }
    }

private:
    template<typename Vector3>
    void sunlight_dirs(
            unsigned int n_dirs,
            std::vector<Vector3> &light_wi,
            Sampler<typename Vector3::Scalar> &sampler) const {

        namespace math = internal::math;
        using Scalar = typename Vector3::Scalar;

        assert(std::abs(noon.norm()-1.) < 1e-6);
        assert(std::abs(east.dot(noon)) < 1e-6);

        light_wi.resize(n_dirs);
        for (int ii = 0; ii < n_dirs; ++ii) {
            const auto theta = math::Pi<Scalar>*(Scalar(ii)+sampler.sample())/Scalar(n_dirs);
            const auto cos_theta = std::cos(theta);
            const auto sin_theta = std::sin(theta);
            light_wi[ii] = cos_theta*east + sin_theta*noon;
        }
    }

    template<typename Vector3>
    void direct_to_point(
            unsigned int n_dirs,
            const Vector3 &pos,
            std::vector<Vector3> &sensor_wo,
            Sampler<typename Vector3::Scalar> &sampler) const {

        namespace math = internal::math;
        using Scalar = typename Vector3::Scalar;

        sensor_wo.resize(n_dirs);

        for (int ii = 0; ii < n_dirs; ++ii) {
            sensor_wo[ii] = (target-pos).normalized();
        }
    }

    template<typename Vector3>
    void direct_to_line(
            unsigned int n_dirs,
            const Vector3 &pos,
            std::vector<Vector3> &sensor_wo,
            Sampler<typename Vector3::Scalar> &sampler) const {

        namespace math = internal::math;
        using Scalar = typename Vector3::Scalar;

        const Vector3 line = decltype(line)::UnitX();
        const auto target_on_line = target + (target-pos).dot(line)*line;

        sensor_wo.resize(n_dirs);

        for (int ii = 0; ii < n_dirs; ++ii) {
            sensor_wo[ii] = (target_on_line-pos).normalized();
        }
    }

};


} // namespace stealth