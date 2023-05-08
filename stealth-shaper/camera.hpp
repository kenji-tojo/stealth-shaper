#pragma once

#include <memory>

#include "internal/math.hpp"
#include "ray.hpp"


namespace stealth {

namespace math = internal::math;

class Camera {
public:
    Eigen::Vector3f position{0.f,100.f,100.f};
    Eigen::Vector3f center{0.f,0.f,0.f};
    float fov = 60.f;

    void look_at() { m_to = (center-position).normalized(); }
    void look_at(const Eigen::Vector3f &target) { center = target; look_at(); }

    [[nodiscard]] Ray spawn_ray(float x_ndc, float y_ndc) const {
        using namespace Eigen;
        const Vector3f right = m_to.cross(m_up).normalized();
        const Vector3f up = right.cross(m_to).normalized();

        float fov_rad = float(M_PI) * fov / 180.f;
        float t = std::tan(.5f*fov_rad);
        float screen_x = t * x_ndc;
        float screen_y = t * y_ndc;

        Vector3f dir = m_to + screen_x*right + screen_y*up;
        dir.normalize();

        return {position, dir};
    }

private:
    Eigen::Vector3f m_up = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f m_to = -Eigen::Vector3f::UnitX();

};

} // namespace stealth