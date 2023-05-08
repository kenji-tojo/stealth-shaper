#pragma once

#include <tuple>
#include <random>

#include "internal/math.hpp"


namespace stealth {

template<typename T>
class Sampler {
public:
    Sampler() : rd(), gen(rd()), dis(T(0), T(1)) { static_assert(std::is_floating_point_v<T>); }

    void set_seed(unsigned int seed) { gen = std::mt19937(seed); }

    T sample() { return dis(gen); }
private:
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<T> dis;
};

template<typename T>
std::pair<T,T> sample_disc(Sampler<T> &sampler) {
    constexpr auto Pi2 = internal::math::Pi2<T>;
    T theta = sampler.sample() * Pi2;
    T r = std::sqrt(sampler.sample());
    return make_pair(r*std::cos(theta), r*std::sin(theta));
}

template<typename T>
std::pair<T,T> sample_tri(Sampler<T> &sampler) {
    auto a = std::sqrt(sampler.sample());
    auto b = sampler.sample() * a;
    return std::make_pair(T(1)-a, b);
}

template<typename Vector3>
void sample_tri(
        const Vector3 &p0,
        const Vector3 &p1,
        const Vector3 &p2,
        Vector3 &out,
        Sampler<typename Vector3::Scalar> &sampler) {
    using Scalar = typename Vector3::Scalar;
    auto [u,v] = sample_tri(sampler);
    out = u*p0 + v*p1 + (Scalar(1.)-u-v)*p2;
}

} // namespace stealth
