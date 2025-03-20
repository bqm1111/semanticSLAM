#pragma once
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <string>
#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"
#include <queue>

namespace voxblox
{
    class SemTsdfServer
    {
    public:
        SemTsdfServer(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
        ~SemTsdfServer() {}

        void integratePointcloud(const sensor_msgs::PointCloud2::Ptr &pointcloud_msg_in);
        bool getNextPointcloudFromQueue(std::queue<sensor_msgs::PointCloud2::Ptr> *queue,
                                        sensor_msgs::PointCloud2::Ptr *pointcloud_msg, Transformation *T_G_C);
        void processPointCloudMessageAndInsert(
            const sensor_msgs::PointCloud2::Ptr &pointcloud_msg,
            const Transformation &T_G_C, const bool is_freespace_pointcloud);
        void publishPointclouds();

    protected:
        ros::NodeHandle nh_;
        ros::NodeHandle nh_private_;
        /// What output information to publish
        bool publish_pointclouds_on_update_;
        bool publish_slices_;
        bool publish_pointclouds_;
        bool publish_tsdf_map_;

        // Data subscriber
        ros::Subscriber pointcloud_sub_;
        tf::TransformListener tf_listener_;
        std::string world_frame_id_;
        /**
         * Transformer object to keep track of either TF transforms or messages from
         * a transform topic.
         */

        Transformer transformer_;

        /// Subscriber settings.
        int pointcloud_queue_size_;
        int num_subscribers_tsdf_map_;
        std::queue<sensor_msgs::PointCloud2::Ptr> pointcloud_queue_;
        std::queue<sensor_msgs::PointCloud2::Ptr> freespace_pointcloud_queue_;
        /// Will throttle to this message rate.
        ros::Duration min_time_between_msgs_;

        // Last message times for throttling input.
        ros::Time last_msg_time_ptcloud_;
        ros::Time last_msg_time_freespace_ptcloud_;
        // Timers.
        ros::Timer update_mesh_timer_;
        ros::Timer publish_map_timer_;

        bool verbose_;
    };
}
