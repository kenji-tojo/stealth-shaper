cmake_minimum_required(VERSION 3.20)
project(stealth LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_subdirectory(stealth-shaper)

add_executable(stealth-headless headless.cpp)
target_include_directories(stealth-headless PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ext
)
target_link_libraries(stealth-headless PRIVATE
    Eigen3::Eigen
    stealth-shaper
)