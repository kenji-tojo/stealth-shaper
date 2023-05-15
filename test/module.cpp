#include <iostream>
#include <memory>
#include <vector>

#include "stealth-shaper/internal/timer.hpp"

#include "stealth-shaper/trimesh.h"
#include "stealth-shaper/plane.h"
#include "stealth-shaper/camera.hpp"
#include "stealth-shaper/renderer.hpp"
#include "stealth-shaper/visualizer.hpp"
#include "stealth-shaper/optimizer.hpp"
#include "stealth-shaper/geometry.hpp"

#include <Eigen/Geometry>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>


namespace nb = nanobind;
using namespace nb::literals;

namespace st = stealth;

namespace {

using Vector3 = Eigen::Vector3f;
using Matrix = Eigen::MatrixXf;

}

class NbTensor {
public:
    std::shared_ptr<st::Tensor<Matrix>> tensor;

    [[nodiscard]] nb::ndarray<nb::numpy, float> numpy() const {
        if (!tensor) return {};
        size_t ndim = 2;
        size_t shape[2] = {tensor->rows(), tensor->cols()};
        return {tensor->mat->data(), ndim, shape};
    }

    void descent(float alpha) { tensor->descent(alpha); }

    [[nodiscard]] unsigned int rows() const { return tensor->rows(); }
    [[nodiscard]] unsigned int cols() const { return tensor->cols(); }
};


class NbSGD: public st::SGD<st::Tensor<Matrix>> {
public:
    using SGD = st::SGD<st::Tensor<Matrix>>;

    float alpha = 1.f;
    void add_parameters(const NbTensor &parameters) {
        SGD::add_parameters(parameters.tensor);
    }
    void step() { SGD::descent(alpha); }
};


class NbAdam: public st::Adam<st::Tensor<Matrix>> {
public:
    using Adam = st::Adam<st::Tensor<Matrix>>;

    float alpha = 1.f;
    void add_parameters(const NbTensor &parameters) {
        Adam::add_parameters(parameters.tensor);
    }
    void step() { Adam::descent(alpha); }
};


class NbTriMesh {
public:
    NbTriMesh(const char *path) { mesh = std::make_shared<st::TriMesh>(std::string{path}); }

    void print_info() const { assert(mesh); mesh->print_info(); }
    void save(const char *path) { mesh->write_obj(path); }

    [[nodiscard]] nb::ndarray<nb::numpy, float> normalized_areas() const {
        const auto &A = mesh->A;
        size_t size = A.size();
        auto data = new float [size];
        for (int ii = 0; ii < size; ++ii) data[ii] = A[ii];
        size_t shape[1] = {size};
        size_t ndim = 1;
        return {data, ndim, shape};
    }

    [[nodiscard]] NbTensor positions_tensor() const {
        NbTensor positions;
        positions.tensor = std::make_shared<st::TriMesh::TensorV>(mesh->V_shared);
        return positions;
    }

    [[nodiscard]] NbTensor normals_tensor() const {
        NbTensor normals;
        normals.tensor = std::make_shared<st::TriMesh::TensorNXY>(mesh->triN_shared);
        return normals;
    }

    [[nodiscard]] nb::ndarray<nb::numpy, float> V() const {
        size_t nV = mesh->V().rows();
        size_t shape[2] = {nV, 3};
        auto data = new float[shape[0]*shape[1]];
        for (int ii = 0; ii < nV; ++ii) {
            data[3*ii+0] = mesh->V()(ii,0);
            data[3*ii+1] = mesh->V()(ii,1);
            data[3*ii+2] = mesh->V()(ii,2);
        }
        return {data, /*ndim=*/2, shape};
    }

    [[nodiscard]] nb::ndarray<nb::numpy, float> refV() const {
        size_t nV = mesh->refV.rows();
        size_t shape[2] = {nV, 3};
        auto data = new float[shape[0]*shape[1]];
        for (int ii = 0; ii < nV; ++ii) {
            data[3*ii+0] = mesh->refV(ii,0);
            data[3*ii+1] = mesh->refV(ii,1);
            data[3*ii+2] = mesh->refV(ii,2);
        }
        return {data, /*ndim=*/2, shape};
    }

    [[nodiscard]] nb::ndarray<nb::numpy, int> F() const {
        size_t nF = mesh->F().rows();
        size_t shape[2] = {nF, 3};
        auto data = new int[shape[0]*shape[1]];
        for (int ii = 0; ii < nF; ++ii) {
            data[3*ii+0] = mesh->F()(ii,0);
            data[3*ii+1] = mesh->F()(ii,1);
            data[3*ii+2] = mesh->F()(ii,2);
        }
        return {data, /*ndim=*/2, shape};
    }

    [[nodiscard]] unsigned int get_n_tris() const { assert(mesh); return mesh->F().rows(); }
    [[nodiscard]] float get_avg_edge_length() const { assert(mesh); return mesh->avg_edge_length; }
    [[nodiscard]] float get_total_area() const { assert(mesh); return mesh->totalA; }

    void update_positions() const { mesh->update_positions(); }

    void rotate(double angleX, double angleY, double angleZ) {
        using namespace Eigen;
        constexpr auto Pi = st::internal::math::Pi<double>;
        Matrix3d rot;
        rot = AngleAxisd(Pi*angleX/180.0, Vector3d::UnitX())
              * AngleAxisd(Pi*angleY/180.0, Vector3d::UnitY())
              * AngleAxisd(Pi*angleZ/180.0, Vector3d::UnitZ());
        auto V = mesh->refV;
        auto F = mesh->F();
        for (int ii = 0; ii < V.rows(); ++ii) {
            V.row(ii) = (rot * V.row(ii).transpose()).transpose();
        }
        mesh = std::make_shared<st::TriMesh>(std::move(V), std::move(F));
    }

    void scale_from_target_diameter(double diameter) {
        auto V = mesh->refV;
        auto F = mesh->F();
        V *= diameter / st::geometry::diameter(V);
        mesh = std::make_shared<st::TriMesh>(std::move(V), std::move(F));
    }

    std::shared_ptr<st::TriMesh> mesh;
};


class NbPlane {
public:
    NbPlane() = default;

    void print_info() const {
        assert(plane);
        plane->print_info();
    }

    void set_normal(float x, float y, float z) const {
        assert(plane);
        plane->normal[0] = x;
        plane->normal[1] = y;
        plane->normal[2] = z;
        plane->normal.normalize();
        plane->update_frame();
    }

    void set_center(float x, float y, float z) const {
        assert(plane);
        plane->center[0] = x;
        plane->center[1] = y;
        plane->center[2] = z;
    }

    void set_scale(float x, float y) const {
        assert(plane);
        plane->scale[0] = x;
        plane->scale[1] = y;
    }

    void set_normal_map(nb::ndarray<float, nb::numpy, nb::shape<nb::any, nb::any, 3>> &normal_map) const {
        assert(plane);
        unsigned int tex_res_x = normal_map.shape(0);
        unsigned int tex_res_y = normal_map.shape(1);

        std::vector<float> normals(tex_res_x * tex_res_y * 3);

        for (int t0 = 0; t0 < tex_res_x; ++t0) {
            for (int t1 = 0; t1 < tex_res_y; ++t1) {
                unsigned int index = tex_res_y * t0 + t1;
                normals[index * 3 + 0] = normal_map(t0, t1, 0);
                normals[index * 3 + 1] = normal_map(t0, t1, 1);
                normals[index * 3 + 2] = normal_map(t0, t1, 2);
            }
        }

        plane->set_normal_map(normals, tex_res_x, tex_res_y);
    }

    [[nodiscard]] NbTensor normal_map_tensor() const {
        assert(plane->normal_map);
        NbTensor normal_map;
        normal_map.tensor = std::make_shared<st::Plane::TensorNXY>(plane->normal_map);
        return normal_map;
    }

    std::shared_ptr<st::Plane> plane = std::make_shared<st::Plane>();
};


class NbPhongBSDF {
public:
    using PhongBSDF = st::PhongBSDF<Vector3>;

    void set_n(unsigned int n) const { assert(bsdf); bsdf->set_n(n); }
    [[nodiscard]] unsigned int get_n() const { assert(bsdf); return bsdf->get_n(); }

    void set_kd(float kd) const { assert(bsdf); bsdf->kd = kd; }
    [[nodiscard]] float get_kd() const { assert(bsdf); return bsdf->kd; }

    void set_albedo(float albedo) const { assert(bsdf); bsdf->albedo = albedo; }
    [[nodiscard]] float get_albedo() const { assert(bsdf); return bsdf->albedo; }

    std::shared_ptr<PhongBSDF> bsdf = std::make_shared<PhongBSDF>();
};


class NbCamera: public st::Camera {
public:
    void set_position(float x, float y, float z) {
        position.x() = x;
        position.y() = y;
        position.z() = z;
    }

    [[nodiscard]] nb::tuple get_position() const {
        const auto &p = position;
        return nb::make_tuple(p.x(), p.y(), p.z());
    }

    void set_center(float x, float y, float z) {
        center.x() = x;
        center.y() = y;
        center.z() = z;
    }

    void look_at_center() { Camera::look_at(); }

    void look_at(float x, float y, float z) {
        Camera::look_at(Eigen::Vector3f{x, y, z});
    }
};


class NbScene: public st::Scene<Vector3> {
public:
    using Scene = Scene<Vector3>;

    void add_object(const NbTriMesh &obj) {
        Scene::add_object(obj.mesh);
    }
    void add_object(const NbTriMesh &obj, const NbPhongBSDF &phong) {
        Scene::add_object(obj.mesh, phong.bsdf);
    }

    void add_object(const NbPlane &obj) {
        Scene::add_object(obj.plane);
    }
    void add_object(const NbPlane &obj, const NbPhongBSDF &phong) {
        Scene::add_object(obj.plane, phong.bsdf);
    }

    void set_colors(nb::ndarray<nb::numpy, float, nb::shape<nb::any, 3>> &colors_) {
        colors.resize(colors_.shape(0));
        for (int ii = 0; ii < colors_.shape(0); ++ii) {
            colors[ii][0] = colors_(ii,0);
            colors[ii][1] = colors_(ii,1);
            colors[ii][2] = colors_(ii,2);
        }
    }

    void reset() { Scene::clear(); }
};


class NbRenderer: public st::Renderer {
public:
    unsigned int resolution = 64;
    unsigned int channels = 4;

    nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any, nb::any>> render(const NbScene &scene, const NbCamera &camera) {
        using Image = nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any, nb::any>>;

        size_t shape[3] = {resolution, resolution, channels};
        auto data = new float[shape[0] * shape[1] * shape[2]];

        auto image = Image{data, /*ndim=*/3, shape};

        using namespace std;

        cout << "rendering: image size = "
             << image.shape(0) << "x"
             << image.shape(1) << "x"
             << image.shape(2) << endl;

        st::internal::Timer timer;

        st::Renderer::render(scene, camera, image);

        timer.stop();
        cout << "...done: elapsed time = " << timer.elapsed_sec() << " sec" << endl << endl;

        return image;
    }

    void adjoint(
            nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any>> &image_adj,
            const NbScene &scene,
            const NbCamera &camera,
            const NbPlane &obj,
            NbTensor &normal_map) {
        assert(image_adj.ndim() >= 2);
        assert(image_adj.shape(0)==resolution && image_adj.shape(1)==resolution);

        unsigned int n_threads = std::thread::hardware_concurrency();
        std::vector<st::Plane::GradAccN> GAN_pool;
        for (int ii = 0; ii < n_threads; ++ii) {
            GAN_pool.emplace_back(normal_map.tensor, obj.plane->obj_id);
        }

        using namespace std;

        cout << "adjoint rendering" << endl;
        st::internal::Timer timer;

        st::Renderer::adjoint(image_adj, scene, camera, GAN_pool);

        timer.stop();
        cout << "...done: elapsed time = " << timer.elapsed_sec() << " sec" << endl << endl;
    }

    void set_light_wi(float x, float y, float z) {
        this->light.wi[0] = x;
        this->light.wi[1] = y;
        this->light.wi[2] = z;
        this->light.wi.normalize();
    }

    [[nodiscard]] nb::tuple get_light_wi() const {
        const auto &wi = this->light.wi;
        return nb::make_tuple(wi[0], wi[1], wi[2]);
    }

    void set_light_Li(float Li) { this->light.Li = Li; }
    [[nodiscard]] float get_light_Li() const { return this->light.Li; }
};


class NbVisualizer: public st::Visualizer {
public:
    unsigned int resolution = 64;
    unsigned int channels = 4;

    void set_vertex_colors(nb::ndarray<nb::numpy, float, nb::shape<nb::any, 3>> &colors) {
        this->vertex_colors.resize(colors.shape(0));
        for (int ii = 0; ii < colors.shape(0); ++ii) {
            this->vertex_colors[ii][0] = colors(ii,0);
            this->vertex_colors[ii][1] = colors(ii,1);
            this->vertex_colors[ii][2] = colors(ii,2);
        }
    }

    void set_face_colors(nb::ndarray<nb::numpy, float, nb::shape<nb::any, 3>> &colors) {
        this->face_colors.resize(colors.shape(0));
        for (int ii = 0; ii < colors.shape(0); ++ii) {
            this->face_colors[ii][0] = colors(ii,0);
            this->face_colors[ii][1] = colors(ii,1);
            this->face_colors[ii][2] = colors(ii,2);
        }
    }

    nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any, nb::any>> render(
            const NbTriMesh &mesh,
            const NbScene &scene,
            const NbCamera &camera) {
        using Image = nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any, nb::any>>;

        size_t shape[3] = {resolution, resolution, channels};
        auto data = new float [shape[0]*shape[1]*shape[2]];

        auto image = Image{data, /*ndim=*/3, shape};

        using namespace std;

        cout << "rendering: image size = "
             << image.shape(0) << "x"
             << image.shape(1) << "x"
             << image.shape(2) << endl;

        st::internal::Timer timer;

        st::Visualizer::render(*mesh.mesh, scene, camera, image);

        timer.stop();
        cout << "...done: elapsed time = " << timer.elapsed_sec() << " sec" << endl << endl;

        return image;
    }
};



NB_MODULE(stealth, m) {

#define DEF_PROP_RW(class_, name) .def_prop_rw(#name, &class_::get_##name, &class_::set_##name)

    nb::class_<NbTensor>(m, "Tensor")
            .def(nb::init<>())
            .def("descent", &NbTensor::descent)
            .def("numpy", &NbTensor::numpy, nb::rv_policy::reference);

    nb::class_<NbSGD>(m, "SGD")
            .def(nb::init<>())
            .def("add_parameters", nb::overload_cast<const NbTensor &>(&NbSGD::add_parameters))
            .def_rw("alpha", &NbSGD::alpha)
            .def_rw("beta", &NbSGD::beta)
            .def_rw("iter_max", &NbSGD::iter_max)
            .def("step", &NbSGD::step);

    nb::class_<NbAdam>(m, "Adam")
            .def(nb::init<>())
            .def("add_parameters", nb::overload_cast<const NbTensor &>(&NbAdam::add_parameters))
            .def_rw("alpha", &NbAdam::alpha)
            .def_rw("beta1", &NbAdam::beta1)
            .def_rw("beta2", &NbAdam::beta2)
            .def_rw("eps", &NbAdam::eps)
            .def_rw("iter_max", &NbAdam::iter_max)
            .def("step", &NbAdam::step);

    nb::class_<NbTriMesh>(m, "TriMesh")
            .def(nb::init<const char *>())
            .def("print_info", &NbTriMesh::print_info)
            .def("save", &NbTriMesh::save)
            .def("normalized_areas", &NbTriMesh::normalized_areas, nb::rv_policy::take_ownership)
            .def("positions_tensor", &NbTriMesh::positions_tensor)
            .def("normals_tensor", &NbTriMesh::normals_tensor)
            .def("V", &NbTriMesh::V, nb::rv_policy::take_ownership)
            .def("refV", &NbTriMesh::refV, nb::rv_policy::take_ownership)
            .def("F", &NbTriMesh::F, nb::rv_policy::take_ownership)
            .def("update_positions", &NbTriMesh::update_positions)
            .def("rotate", &NbTriMesh::rotate)
            .def("scale_from_target_diameter", &NbTriMesh::scale_from_target_diameter)
            .def_prop_ro("n_tris", &NbTriMesh::get_n_tris)
            .def_prop_ro("avg_edge_length", &NbTriMesh::get_avg_edge_length)
            .def_prop_ro("total_area", &NbTriMesh::get_total_area);

    nb::class_<NbPlane>(m, "Plane")
            .def(nb::init<>())
            .def("print_info", &NbPlane::print_info)
            .def("set_center", &NbPlane::set_center)
            .def("set_normal", &NbPlane::set_normal)
            .def("normal_map_tensor", &NbPlane::normal_map_tensor)
            .def("set_scale", &NbPlane::set_scale)
            .def("set_normal_map", &NbPlane::set_normal_map);

    nb::class_<NbPhongBSDF>(m, "PhongBSDF")
            .def(nb::init<>())
            DEF_PROP_RW(NbPhongBSDF, n)
            DEF_PROP_RW(NbPhongBSDF, kd)
            DEF_PROP_RW(NbPhongBSDF, albedo);

    nb::class_<NbCamera>(m, "Camera")
            .def(nb::init<>())
            .def_rw("fov", &NbCamera::fov)
            .def("set_position", &NbCamera::set_position)
            .def("set_center", &NbCamera::set_center)
            .def_prop_ro("position", &NbCamera::get_position)
            .def("look_at_center", &NbCamera::look_at_center)
            .def("look_at", &NbCamera::look_at);

    nb::class_<NbScene>(m, "Scene")
            .def(nb::init<>())
            .def("add_object", nb::overload_cast<const NbTriMesh &>(&NbScene::add_object))
            .def("add_object", nb::overload_cast<const NbTriMesh &, const NbPhongBSDF &>(&NbScene::add_object))
            .def("add_object", nb::overload_cast<const NbPlane &>(&NbScene::add_object))
            .def("add_object", nb::overload_cast<const NbPlane &, const NbPhongBSDF &>(&NbScene::add_object))
            .def("reset", &NbScene::reset)
            .def("set_colors", &NbScene::set_colors);

    nb::class_<NbRenderer>(m, "Renderer")
            .def(nb::init<>())
            .def_rw("spp", &NbRenderer::spp)
            .def_rw("resolution", &NbRenderer::resolution)
            .def_rw("channels", &NbRenderer::channels)
            .def_rw("depth", &NbRenderer::depth)
            .def_prop_rw("light_Li", &NbRenderer::get_light_Li, &NbRenderer::set_light_Li)
            .def("set_light_wi", &NbRenderer::set_light_wi)
            .def_rw("clip_max", &NbRenderer::clip_max)
            .def_prop_ro("light_wi", &NbRenderer::get_light_wi)
            .def("render", &NbRenderer::render, nb::rv_policy::take_ownership)
            .def("adjoint", &NbRenderer::adjoint);

    nb::class_<NbVisualizer>(m, "Visualizer")
            .def(nb::init<>())
            .def_rw("spp", &NbVisualizer::spp)
            .def_rw("type", &NbVisualizer::type)
            .def_rw("resolution", &NbVisualizer::resolution)
            .def_rw("channels", &NbVisualizer::channels)
            .def("set_vertex_colors", &NbVisualizer::set_vertex_colors)
            .def("set_face_colors", &NbVisualizer::set_face_colors)
            .def("render", &NbVisualizer::render, nb::rv_policy::take_ownership);

} // NB_MODULE(stealth, m)
