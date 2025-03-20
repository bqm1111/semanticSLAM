#pragma once
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
namespace voxblox
{
#define NUM_CLASSES 41
    struct PointXYZRGBSemantic
    {
        PCL_ADD_POINT4D;
        PCL_ADD_RGB;
        uint8_t label;
        float confidence;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } EIGEN_ALIGN16;

    struct PointXYZRGBSemanticBayesian
    {
        PCL_ADD_POINT4D;
        PCL_ADD_RGB;
        uint8_t label;
        float confidence[NUM_CLASSES];
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } EIGEN_ALIGN16;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(
    voxblox::PointXYZRGBSemantic,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(uint32_t, label, label)(float, confidence, confidence))
POINT_CLOUD_REGISTER_POINT_STRUCT(
    voxblox::PointXYZRGBSemanticBayesian,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(uint32_t, label, label)(float[NUM_CLASSES], confidence, confidence))