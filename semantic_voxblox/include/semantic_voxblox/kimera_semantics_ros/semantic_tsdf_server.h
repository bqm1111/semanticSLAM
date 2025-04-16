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
 * @file   semantic_tsdf_server.h
 * @brief  Semantic TSDF Server to interface with ROS
 * @author Antoni Rosinol
 */
#pragma once

#include <ros/ros.h>

#include <voxblox_ros/tsdf_server.h>

#include "semantic_voxblox/semantic_voxel.h"
#include "semantic_voxblox/semantic_integrator_base.h"
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <voxblox/core/common.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>
#include <minkindr_conversions/kindr_msg.h>

namespace kimera
{
	class SemanticTsdfServer : public vxb::TsdfServer
	{
	public:
		SemanticTsdfServer(const ros::NodeHandle &nh,
						   const ros::NodeHandle &nh_private);

		SemanticTsdfServer(const ros::NodeHandle &nh,
						   const ros::NodeHandle &nh_private,
						   const vxb::TsdfMap::Config &config,
						   const vxb::TsdfIntegratorBase::Config &integrator_config,
						   const vxb::MeshIntegratorConfig &mesh_config);

		virtual ~SemanticTsdfServer() = default;

	protected:
		void insertPointcloud(const sensor_msgs::PointCloud2::ConstPtr &pointcloud_msg_in, const geometry_msgs::TransformStamped::ConstPtr &camera_pose_msg);
		voxblox::Transformation odomMsgToVoxbloxTransform(const geometry_msgs::TransformStamped &pose_stamped);

		// Configs.
		SemanticIntegratorBase::SemanticConfig semantic_config_;
		// Layers.
		std::unique_ptr<vxb::Layer<SemanticVoxel>> semantic_layer_;
		message_filters::Subscriber<sensor_msgs::PointCloud2> semantic_pointcloud_sub_;
		// message_filters::Subscriber<nav_msgs::Odometry> camera_pose_sub_;
		message_filters::Subscriber<geometry_msgs::TransformStamped> camera_pose_sub_;

		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, geometry_msgs::TransformStamped> SyncPolicy;

		std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;  // Use a shared pointer
	};
} // Namespace kimera
