#ifndef NORMAL_DRIVEN_PRECOMPUTATION_H
#define NORMAL_DRIVEN_PRECOMPUTATION_H

#include <iostream>
#include <ctime>
#include <vector>
#include <Eigen/Core>

// include libigl functions
#include <igl/cotmatrix_entries.h>
#include <igl/doublearea.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/columnize.h>
#include <igl/slice.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/sum.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/massmatrix.h>

// include project functions
#include "normal_driven_data.h"
#include "print_sparse_mat.h"
#include "sp_cwiseInverse.h"

void normal_driven_precomputation(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const int & type, // 0: vertex arap, 1: face arap
    normal_driven_data & data);

void normal_driven_precomputation_vertex(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    normal_driven_data & data);

void normal_driven_precomputation_face(
	const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    normal_driven_data & data);
#endif