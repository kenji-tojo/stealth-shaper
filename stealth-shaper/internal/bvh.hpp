#pragma once

#include <iostream>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include "./math.hpp"
#include "../ray.hpp"

namespace bvhv2 {
namespace {

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

inline Ray adapt_ray(const stealth::Ray &ray) {
    const auto &o = ray.org;
    const auto &d = ray.dir;
    return {/*org=*/Vec3(o[0],o[1],o[2]), /*dir=*/Vec3(d[0],d[1],d[2]), ray.tmin, ray.tmax};
}

inline std::vector<Tri> adapt_tris(const Eigen::MatrixXf &V, const Eigen::MatrixXi &F) {
    std::vector<bvhv2::Tri> tris(F.rows());
    for (int ii = 0; ii < F.rows(); ++ii) {
        const Eigen::Vector3f &p0 = V.row(F(ii,0));
        const Eigen::Vector3f &p1 = V.row(F(ii,1));
        const Eigen::Vector3f &p2 = V.row(F(ii,2));
        tris[ii] = bvhv2::Tri(
                bvhv2::Vec3(p0.x(),p0.y(),p0.z()),
                bvhv2::Vec3(p1.x(),p1.y(),p1.z()),
                bvhv2::Vec3(p2.x(),p2.y(),p2.z())
        );
    }
    return tris;
}

} // namespace
} // namespace bvhv2

namespace stealth::internal {

template<bool should_permute_ = false>
class Bvh {
public:
    Bvh(const Eigen::MatrixXf &V, const Eigen::MatrixXi &F) {
        const auto tris = bvhv2::adapt_tris(V, F);

        std::cout << "bvh: number of tris = " << tris.size() << std::endl;

        bvh::v2::ThreadPool thread_pool;
        bvh::v2::ParallelExecutor executor(thread_pool);

        // Get triangle centers and bounding boxes (required for BVH builder)
        std::vector<bvhv2::BBox> bboxes(tris.size());
        std::vector<bvhv2::Vec3> centers(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                bboxes[i]  = tris[i].get_bbox();
                centers[i] = tris[i].get_center();
            }
        });

        typename bvh::v2::DefaultBuilder<bvhv2::Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<bvhv2::Node>::Quality::High;
        bvh = std::make_unique<bvhv2::Bvh>(bvh::v2::DefaultBuilder<bvhv2::Node>::build(thread_pool, bboxes, centers, config));

        // This precomputes some data to speed up traversal further.
        precomputed_tris.resize(tris.size());
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                auto j = should_permute_ ? bvh->prim_ids[i] : i;
                precomputed_tris[i] = tris[j];
            }
        });
    }

    bool raycast(const Ray &_ray, size_t &tri_id, float &dist, Eigen::Vector3f &nrm) {
        assert(bvh);

        auto ray = bvhv2::adapt_ray(_ray);

        static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
        static constexpr size_t stack_size = 64;
        static constexpr bool use_robust_traversal = false;

        auto prim_id = invalid_id;
        bvhv2::Scalar u, v; // barycentric coords

        // Traverse the BVH and get the u, v coordinates of the closest intersection.
        bvh::v2::SmallStack<bvhv2::Bvh::Index, stack_size> stack;
        bvh->intersect<false, use_robust_traversal>(ray, bvh->get_root().index, stack,
                                                    [&] (size_t begin, size_t end) {
                                                        for (size_t i = begin; i < end; ++i) {
                                                            size_t j = should_permute_ ? i : bvh->prim_ids[i];
                                                            if (auto hit = precomputed_tris[j].intersect(ray)) {
                                                                prim_id = i;
                                                                std::tie(u, v) = *hit;
                                                            }
                                                        }
                                                        return prim_id != invalid_id;
                                                    });

        bool intersect = (prim_id != invalid_id);

        if (intersect) {
            const auto &tri = precomputed_tris[should_permute_ ? prim_id : bvh->prim_ids[prim_id]];
            nrm = -Eigen::Vector3f{tri.n.values};
            nrm.normalize();
            tri_id = bvh->prim_ids[prim_id];
            dist = ray.tmax;
        }

        return intersect;
    }

protected:
    std::unique_ptr<bvhv2::Bvh> bvh;
    std::vector<bvhv2::PrecomputedTri> precomputed_tris;

};

} // namespace stealth
