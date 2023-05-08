#pragma once

#include <memory>

#include "internal/math.hpp"
#include "object.hpp"
#include "normal.hpp"
#include "tensor.hpp"


namespace stealth {

class Plane: public Object {
public:
    using Vector2 = Eigen::Vector2f;
    using Vector3 = Eigen::Vector3f;
    using MatrixN = Eigen::MatrixXf;
    using TensorNXY = TensorUnit3XY<MatrixN, Vector3>;
    using GradAccN = GradientAccumulatorNormal<MatrixN, Vector3>;

    Plane();
    ~Plane();

    void raycast(const Ray &ray, Hit &hit) const override;

    void print_info() const;

    Vector3 center = Vector3::Zero();
    Vector3 normal = Vector3::UnitZ();

    Vector3 b1 = Vector3::UnitX();
    Vector3 b2 = Vector3::UnitY();

    Vector2 scale = Vector2::Ones();

    std::shared_ptr<MatrixN> normal_map;
    int tex_res[2] = {0,0};

    void set_normal_map(const std::vector<float> &normals, unsigned int tex_res_x, unsigned int tex_res_y) {
        assert(tex_res_x*tex_res_y==normals.size()/3);
        tex_res[0] = tex_res_x;
        tex_res[1] = tex_res_y;
        normal_map = std::make_shared<MatrixN>();
        normal_map->resize(tex_res[0]*tex_res[1], 3);
        auto &nrm = *normal_map;
        for (int ii = 0; ii < normals.size()/3; ++ii) {
            nrm(ii,0) = normals[3*ii+0];
            nrm(ii,1) = normals[3*ii+1];
            nrm(ii,2) = normals[3*ii+2];
            nrm.row(ii).normalize();
        }
    }

    void update_frame() { internal::math::create_local_frame(normal, b1, b2); }

};

} // namespace stealth