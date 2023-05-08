#pragma once


namespace stealth {

template<typename Vector3>
struct Light {
public:
    using Scalar = typename Vector3::Scalar;
    Vector3 wi = Vector3::UnitZ();
    Scalar Li = Scalar(1.);
};

} // namespace stealth