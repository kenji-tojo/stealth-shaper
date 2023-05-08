#pragma once

#include <vector>
#include <memory>

#include "internal/math.hpp"
#include "object.hpp"
#include "bsdf.hpp"


namespace stealth {

template<typename Vector3>
class Scene  {
public:
    void add_object(
            const std::shared_ptr<Object> &object,
            const std::shared_ptr<BSDF<Vector3>> &bsdf = nullptr) {
        if (!object)
            return;
        object->obj_id = objects.size();
        object->mat_id = 0;
        if (bsdf) {
            object->mat_id = bsdfs.size();
            bsdfs.push_back(bsdf);
        }
        objects.push_back(object);
    }

    [[nodiscard]] const Object &object(unsigned int index) const { assert(objects[index]); return *objects[index]; }
    [[nodiscard]] const BSDF<Vector3> &bsdf(unsigned int index) const { assert(bsdfs[index]); return *bsdfs[index]; }

    void raycast(const Ray &ray, Hit &hit) const {
        for (const auto &o: objects)
            o->raycast(ray, hit);
    }

    void clear() { objects.clear(); bsdfs.resize(1); }

    std::vector<Eigen::Vector3f> colors;

private:
    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<BSDF<Vector3>>> bsdfs = {std::make_shared<BSDF<Vector3>>()};

};

} // namespace stealth