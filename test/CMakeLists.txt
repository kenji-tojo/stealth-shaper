cmake_minimum_required(VERSION 3.20)
project(stealth-test LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../stealth-shaper steatlh-shaper)
target_compile_options(stealth-shaper PRIVATE -fPIC)


find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../ext/nanobind nanobind)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

nanobind_add_module(stealth module.cpp)
target_include_directories(stealth PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../)
target_link_libraries(stealth PRIVATE Eigen3::Eigen stealth-shaper)
