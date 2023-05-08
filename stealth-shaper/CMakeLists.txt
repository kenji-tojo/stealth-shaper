cmake_minimum_required(VERSION 3.20)
project(stealth-shaper LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(BVH_DIR ${CMAKE_CURRENT_LIST_DIR}/ext/bvh/src)


set(LIBIGL_WITH_OPENGL OFF)
set(LIBIGL_WITH_OPENGL_GLFW OFF)
set(LIBIGL_WITH_VIEWER OFF)
set(LIBIGL_WITH_OPENGL_GLFW_IMGUI OFF)
set(LIBIGL_DIR ${CMAKE_CURRENT_LIST_DIR}/ext/libigl)
add_subdirectory(${LIBIGL_DIR})

file(GLOB NORMAL_DRIVEN_SRC ${CMAKE_CURRENT_LIST_DIR}/normal-driven/*.cpp)
file(GLOB ADAPTIVE_MESH_SRC ${CMAKE_CURRENT_LIST_DIR}/adaptive-mesh/*.cpp)

add_library(stealth-shaper
    trimesh.cpp
    plane.cpp
    ${NORMAL_DRIVEN_SRC}
    ${ADAPTIVE_MESH_SRC}
)

target_include_directories(stealth-shaper PRIVATE ${BVH_DIR})
target_include_directories(stealth-shaper PUBLIC ${LIBIGL_DIR}/include)
target_link_libraries(stealth-shaper PRIVATE Eigen3::Eigen)