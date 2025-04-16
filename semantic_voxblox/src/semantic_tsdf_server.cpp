// NOTE: Most code is derived from voxblox: github.com/ethz-asl/voxblox
// Copyright (c) 2016, ETHZ ASL
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of voxblox nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

/**
 * @file   semantic_tsdf_server.cpp
 * @brief  Semantic TSDF Server to interface with ROS
 * @author Antoni Rosinol
 */
#include <ros/ros.h>
#include "semantic_voxblox/kimera_semantics_ros/semantic_tsdf_server.h"
#include <glog/logging.h>
#include <voxblox_ros/ros_params.h>
#include <semantic_voxblox/semantic_tsdf_integrator_factory.h>
#include "semantic_voxblox/kimera_semantics_ros/ros_params.h"
#include <rosconsole/macros_generated.h>

namespace kimera
{
	SemanticTsdfServer::SemanticTsdfServer(const ros::NodeHandle &nh,
										   const ros::NodeHandle &nh_private)
		: SemanticTsdfServer(nh,
							 nh_private,
							 vxb::getTsdfMapConfigFromRosParam(nh_private),
							 vxb::getTsdfIntegratorConfigFromRosParam(nh_private),
							 vxb::getMeshIntegratorConfigFromRosParam(nh_private))
	{
	}
	SemanticTsdfServer::SemanticTsdfServer(
		const ros::NodeHandle &nh,
		const ros::NodeHandle &nh_private,
		const vxb::TsdfMap::Config &config,
		const vxb::TsdfIntegratorBase::Config &integrator_config,
		const vxb::MeshIntegratorConfig &mesh_config)
		: vxb::TsdfServer(nh, nh_private, config, integrator_config, mesh_config),
		  semantic_config_(getSemanticTsdfIntegratorConfigFromRosParam(nh_private)),
		  semantic_layer_(nullptr)
	{
		semantic_pointcloud_sub_.subscribe(nh_, "semantic_pcl/semantic_pcl", 1);
		camera_pose_sub_.subscribe(nh_, "semantic_pcl/camera_pose", 1);
		sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), semantic_pointcloud_sub_, camera_pose_sub_);

		sync_->registerCallback(boost::bind(&SemanticTsdfServer::insertPointcloud, this, _1, _2));

		/// Semantic layer
		semantic_layer_.reset(new vxb::Layer<SemanticVoxel>(
			config.tsdf_voxel_size, config.tsdf_voxels_per_side));
		/// Replace the TSDF integrator by the SemanticTsdfIntegrator
		tsdf_integrator_ =
			SemanticTsdfIntegratorFactory::create(
				getSemanticTsdfIntegratorTypeFromRosParam(nh_private),
				integrator_config,
				semantic_config_,
				tsdf_map_->getTsdfLayerPtr(),
				semantic_layer_.get());
		CHECK(tsdf_integrator_);
	}

	voxblox::Transformation SemanticTsdfServer::odomMsgToVoxbloxTransform(const geometry_msgs::TransformStamped &tf_msg)
	{
		Eigen::Quaternion<float> rotation_f;
		Eigen::Vector3d position_d;
		Eigen::Vector3f position_f;
		
		// Convert ROS message to kindr-compatible data
		tf::quaternionMsgToKindr(tf_msg.transform.rotation, &rotation_f);
		tf::vectorMsgToKindr(tf_msg.transform.translation, &position_d);

		// Cast position to float
		position_f = position_d.cast<float>();

		// Step 2: Create voxblox::Transformation
		voxblox::Transformation T_G_C(rotation_f, position_f);
		return T_G_C;
	}

	void SemanticTsdfServer::insertPointcloud(const sensor_msgs::PointCloud2::ConstPtr &pointcloud_msg_in, const geometry_msgs::TransformStamped::ConstPtr &camera_pose_msg)
	{
		voxblox::Transformation T_G_C;
		T_G_C = odomMsgToVoxbloxTransform(*camera_pose_msg);
		constexpr bool is_freespace_pointcloud = false;

		sensor_msgs::PointCloud2::Ptr pointcloud_msg(new sensor_msgs::PointCloud2(*pointcloud_msg_in));

		processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
										  is_freespace_pointcloud);

		if (publish_pointclouds_on_update_)
		{
			publishPointclouds();
		}
	}
} // Namespace kimera
