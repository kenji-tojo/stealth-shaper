#pragma once

#include <tuple>

#include "internal/math.hpp"
#include "sampler.hpp"


namespace stealth {

namespace math = internal::math;

template<typename Vector3_ = Eigen::Vector3f>
class BSDF {
public:
    using Vector3 = Vector3_;
    using Scalar = typename Vector3::Scalar;

    const char *name = "none";

    BSDF(): BSDF("diffuse") { static_assert(std::is_floating_point_v<Scalar>); }

    Scalar albedo = .5f;

    [[nodiscard]] virtual Scalar eval(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const {
        constexpr auto Pi = math::Pi<Scalar>;
        return albedo * math::max(Scalar(0.), nrm.dot(wi)) / Pi;
    }

    virtual Vector3 dFdN(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const {
        constexpr auto Pi = math::Pi<Scalar>;
        return albedo * Scalar(nrm.dot(wi)>0) * wi / Pi;
    }

    [[nodiscard]] virtual Scalar pdf(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const {
        constexpr auto Pi = math::Pi<Scalar>;
        return std::max(Scalar(0.), nrm.dot(wi)) / Pi; // cosine-weighted hemisphere sampling
    }

    virtual std::tuple<Vector3,Scalar,Scalar> sample_wi(const Vector3 &nrm, const Vector3 &wo, Sampler<Scalar> &sampler) const {
        constexpr auto Pi2 = math::Pi2<Scalar>;

        Vector3 b1, b2;
        math::create_local_frame(nrm, b1, b2);

        auto z = std::sqrt(math::max(Scalar(0.), sampler.sample()));
        auto rxy = std::sqrt(math::max(Scalar(0.), Scalar(1.)-z*z));
        auto phi = Pi2 * sampler.sample();

        Vector3 wi        = z*nrm + rxy*cos(phi)*b1 + rxy*sin(phi)*b2;
        Scalar bsdf_value = eval(nrm, wo, wi);
        Scalar bsdf_pdf   = pdf(nrm, wo, wi);
        return std::make_tuple(wi,bsdf_value,bsdf_pdf);
    }

protected:
    explicit BSDF(const char *_name): name(_name) {}

};


template<typename Vector3_ = Eigen::Vector3f>
class SpecularBSDF: public BSDF<Vector3_> {
public:
    using Vector3 = Vector3_;
    using Scalar = typename Vector3::Scalar;
    using Base = BSDF<Vector3>;

    SpecularBSDF(): Base("specular") {}

    [[nodiscard]] Scalar eval(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        return 0.;
    }

    Vector3 dFdN(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        return Vector3::Zero();
    }

    [[nodiscard]] Scalar pdf(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        return 1;
    }

    std::tuple<Vector3,Scalar,Scalar> sample_wi(const Vector3 &nrm, const Vector3 &wo, Sampler<Scalar> &sampler) const override {
        Vector3 wi        = -wo + Scalar(2.) * wo.dot(nrm) * nrm;
        Scalar bsdf_value = this->albedo * math::max(Scalar(0.), wi.dot(nrm));
        Scalar bsdf_pdf   = 1.;
        return std::make_tuple(wi,bsdf_value,bsdf_pdf);
    }

};


template<typename Vector3_ = Eigen::Vector3f>
class GlossyBSDF: public BSDF<Vector3_> {
public:
    using Vector3 = Vector3_;
    using Scalar = typename Vector3::Scalar;
    using Base = BSDF<Vector3>;

    unsigned int n = 5000;

    GlossyBSDF(): Base("glossy") {}

    [[nodiscard]] Scalar eval(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        constexpr auto Pi2 = math::Pi2<Scalar>;
        Vector3_ r = -wo + Scalar(2.) * nrm.dot(wo) * nrm;
        auto c = math::max(Scalar(0.), r.dot(wi));
        auto w = this->albedo * Scalar(n+2) * std::pow(c, Scalar(n)) / Pi2;
        return w * math::max(Scalar(0.), nrm.dot(wi));
    }

    Vector3 dFdN(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        constexpr auto Pi2 = math::Pi2<Scalar>;

        Vector3 r = -wo + Scalar(2.) * nrm.dot(wo) * nrm;
        auto rdwi = r.dot(wi);
        auto c = math::max(Scalar(0.), rdwi);
        auto w = this->albedo * Scalar(n+2) * std::pow(c, Scalar(n)) / Pi2;
        auto ndwi = nrm.dot(wi);
        Vector3 nrm_adj = w * Scalar(ndwi>0.) * wi;

        auto dw = this->albedo * Scalar(n+2) * Scalar(n) * std::pow(c, Scalar(n-1)) / Pi2;
        nrm_adj += math::max(Scalar(0.), ndwi) * dw * Scalar(rdwi>0.) * Scalar(2.) * (ndwi * wo + nrm.dot(wo) * wi);
        return nrm_adj;
    }

    [[nodiscard]] Scalar pdf(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        constexpr auto Pi2 = math::Pi2<Scalar>;
        Vector3_ r = -wo + Scalar(2.) * nrm.dot(wo) * nrm;
        auto c = math::max(Scalar(0.), r.dot(wi));
        return Scalar(n+1) * std::pow(c, Scalar(n)) / Pi2;
    }

    std::tuple<Vector3,Scalar,Scalar> sample_wi(const Vector3 &nrm, const Vector3 &wo, Sampler<Scalar> &sampler) const override {
        constexpr auto Pi2 = math::Pi2<Scalar>;

        Vector3 r = -wo + Scalar(2.) * nrm.dot(wo) * nrm;
        Vector3 b1, b2;
        math::create_local_frame(r, b1, b2);

        auto z = std::pow(math::max(Scalar(0.), sampler.sample()), Scalar(1.)/Scalar(n+1));
        auto rxy = std::sqrt(math::max(Scalar(0.), Scalar(1.)-z*z));
        auto phi = Pi2 * sampler.sample();

        Vector3 wi        = z*r + rxy*cos(phi)*b1 + rxy*sin(phi)*b2;
        Scalar bsdf_value = eval(nrm, wo, wi);
        Scalar bsdf_pdf   = pdf(nrm, wo, wi);
        return std::make_tuple(wi,bsdf_value,bsdf_pdf);
    }

};


template<typename Vector3_ = Eigen::Vector3f>
class PhongBSDF: public BSDF<Vector3_> {
public:
    using Vector3 = Vector3_;
    using Scalar = typename Vector3::Scalar;
    using Base = BSDF<Vector3>;

    Scalar albedo = .5f;
    Scalar kd = .5f;

    PhongBSDF(): Base("phong") {
        diffuse.albedo = 1.f;
        glossy.albedo = 1.f;
        glossy.n = 5;
    }

    void set_n(unsigned int n) { glossy.n = n; }
    [[nodiscard]] unsigned int get_n() const { return glossy.n; }

    [[nodiscard]] Scalar eval(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        auto kd_ = math::clip(kd, Scalar(0.), Scalar(1.));
        return albedo * (kd_ * diffuse.eval(nrm, wo, wi) + (Scalar(1.)-kd_) * glossy.eval(nrm, wo, wi));
    }

    Vector3 dFdN(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        auto kd_ = math::clip(kd, Scalar(0.), Scalar(1.));
        Vector3 nrm_adj = kd_ * diffuse.dFdN(nrm, wo, wi) + (Scalar(1.)-kd_) * glossy.dFdN(nrm, wo, wi);
        return albedo * nrm_adj;
    }

    [[nodiscard]] Scalar pdf(const Vector3 &nrm, const Vector3 &wo, const Vector3 &wi) const override {
        auto kd_ = math::clip(kd, Scalar(0.), Scalar(1.));
        return kd_ * diffuse.pdf(nrm, wo, wi) + (Scalar(1.)-kd_) * glossy.pdf(nrm, wo, wi);
    }

    std::tuple<Vector3,Scalar,Scalar> sample_wi(const Vector3 &nrm, const Vector3 &wo, Sampler<Scalar> &sampler) const override {
        auto kd_ = math::clip(kd, Scalar(0.), Scalar(1.));

        Vector3 wi;
        Scalar bsdf_value, bsdf_pdf;

        if (sampler.sample() < kd_) {
            std::tie(wi,bsdf_value,bsdf_pdf) = diffuse.sample_wi(nrm, wo, sampler);
            bsdf_pdf *= kd_;
        }
        else {
            std::tie(wi,bsdf_value,bsdf_pdf) = glossy.sample_wi(nrm, wo, sampler);
            bsdf_pdf *= (Scalar(1.)-kd_);
        }

        bsdf_value = eval(nrm, wo, wi);
        return std::make_tuple(wi,bsdf_value,bsdf_pdf);
    }

private:
    BSDF<Vector3> diffuse;
    GlossyBSDF<Vector3> glossy;

};

} // namespace stealth