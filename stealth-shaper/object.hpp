#pragma once

#include "ray.hpp"
#include "hit.hpp"


namespace stealth {

class Object {
public:
    const char *name;
    explicit Object(const char *_name): name(_name) {}

    bool enabled = true;
    int obj_id = 0;
    int mat_id = 0;

    virtual void raycast(const Ray &ray, Hit &hit) const = 0;

};

} // namespace stealth