#pragma once

#include <cstdint>
#include <cmath>

#include <numeric>
#include <vector>

#include <Eigen/Dense>

#ifndef M_PI
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif


namespace stealth::internal::math {

template<typename Scalar> constexpr Scalar Pi = Scalar(M_PI);
template<typename Scalar> constexpr Scalar Pi2 = Scalar(2.*M_PI);

template<typename Scalar>
Scalar max(Scalar a, Scalar b) {
    return a < b ? b : a;
}

template<typename Scalar>
Scalar min(Scalar a, Scalar b) {
    return a < b ? a : b;
}

template<typename Scalar>
Scalar clip(Scalar a, Scalar lo, Scalar hi) {
    return max(lo, min(hi, a));
}

template<typename Vector3, typename Scalar>
void clip3(Vector3 &v, Scalar lo, Scalar hi) {
    v[0] = clip(v[0], lo, hi);
    v[1] = clip(v[1], lo, hi);
    v[2] = clip(v[2], lo, hi);
}

template<typename Scalar>
uint8_t to_u8(Scalar a) {
    return uint8_t(Scalar(255) * clip(a, Scalar(0), Scalar(1)));
}

template<typename Vector3>
void create_local_frame(const Vector3 &nrm, Vector3 &b1, Vector3 &b2) {
    const double sign = nrm.z() >= 0 ? 1 : -1;
    const double a = -1.0 / (sign + nrm.z());
    const double b = nrm.x() * nrm.y() * a;
    b1 = Vector3(1.0 + sign * nrm.x() * nrm.x() * a, sign * b, -sign * nrm.x());
    b2 = Vector3(b, sign + nrm.y() * nrm.y() * a, -nrm.y());

    constexpr double eps = 1e-5;
    assert(std::abs(b1.norm() - 1.) < eps);
    assert(std::abs(b2.norm() - 1.) < eps);
    assert(std::abs(b1.dot(nrm)) < eps);
    assert(std::abs(b2.dot(nrm)) < eps);
    assert(std::abs(b1.dot(b2)) < eps);
}

template<typename Float>
inline Float tone_map_Reinhard(const Float c, const Float burn) {
    static_assert(std::is_floating_point_v<Float>);
    return c * (1.0 + c / (burn * burn)) / (1 + c);
}

template<typename Vector3, typename Float>
inline void Reinhard3(Vector3 &v, const Float burn) {
    v[0] = tone_map_Reinhard(v[0], burn);
    v[1] = tone_map_Reinhard(v[1], burn);
    v[2] = tone_map_Reinhard(v[2], burn);
}

template<typename Float>
inline Float sigmoid(const Float c) {
    static_assert(std::is_floating_point_v<Float>);
    return Float(2.) * math::max<Float>(0., -.5+1./(1.+std::exp(-c)));
}

template<typename Vector3>
inline void sigmoid3(Vector3 &v) {
    v[0] = sigmoid(v[0]);
    v[1] = sigmoid(v[1]);
    v[2] = sigmoid(v[2]);
}

template<typename Vector3>
inline void floor3(Vector3 &v) {
    v[0] = std::floor(v[0]);
    v[1] = std::floor(v[1]);
    v[2] = std::floor(v[2]);
}

template<typename Vector3, typename Scalar>
Vector3 slerp(const Vector3 &a, const Vector3 &b, Scalar t) {
    using namespace Eigen;
    Quaternionf qa;
    Quaternionf qb;
    qa = Quaternionf::Identity();
    qb.setFromTwoVectors(a,b);
    return (qa.slerp(t,qb)) * a;
}

template<typename Scalar, typename IndexType>
void argsort_array(
        const Scalar *data,
        const size_t count,
        std::vector<IndexType> &sorted_indices) {

    static_assert(std::is_integral_v<IndexType>);

    sorted_indices.resize(count);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&data] (size_t ii, size_t jj) {
        return data[ii] > data[jj];
    });
}

} // namespace stealth::math