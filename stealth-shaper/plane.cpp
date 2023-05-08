#include "plane.h"

#include <iostream>


namespace stealth {

Plane::Plane(): Object("plane") {}

Plane::~Plane() = default;

void Plane::print_info() const {
    using namespace std;
    cout << "Plane: "
         << "center = " << center << endl
         << "normal = " << normal << endl
         << "scale = " << scale << endl;
}

void Plane::raycast(const Ray &ray, Hit &hit) const {
    if (!this->enabled) { return; }

    auto nrm = normal;
    if (ray.dir.dot(nrm) > 0.f) { nrm *= -1.f; }

    float ddn = -ray.dir.dot(nrm);
    if (ddn < 1e-8f) { return; /* plane is parallel to the ray */ }

    float dist = nrm.dot(ray.org-center) / ddn;
    if (dist < 1e-8f || dist >= hit.dist) { return; }

    const auto pos = ray.org + dist * ray.dir + 1e-4f * nrm;
    float x = b1.dot(pos-center)/scale[0];
    float y = b2.dot(pos-center)/scale[1];
    if (x < -1.f || x > 1.f) { return; }
    if (y < -1.f || y > 1.f) { return; }

    hit.dist = dist;
    hit.prim_id = 0;
    hit.nrm = nrm;
    hit.pos = pos;
    hit.wo = -ray.dir;
    hit.obj_id = this->obj_id;
    hit.mat_id = this->mat_id;

    if (normal_map && tex_res[0]*tex_res[1]==normal_map->rows()) {
        int t0 = std::floor(float(tex_res[0])*.5f*(x+1.f));
        int t1 = std::floor(float(tex_res[1])*.5f*(y+1.f));
        t0 = internal::math::clip<int>(t0, 0, tex_res[0]-1);
        t1 = internal::math::clip<int>(t1, 1, tex_res[1]-1);
        hit.nrm = normal_map->row(tex_res[1]*t0+t1).transpose();
        hit.prim_id = tex_res[1]*t0+t1;
    }
}

} // namespace stealth