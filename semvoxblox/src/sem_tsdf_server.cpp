#include "semvoxblox/sem_tsdf_server.h"

namespace voxblox
{
    SemTsdfServer::SemTsdfServer(ros::NodeHandle &nh, ros::NodeHandle &nh_private) : nh_(nh), nh_private_(nh_private), transformer_(nh, nh_private)
    {
        pointcloud_sub_ = nh_.subscribe("pointcloud", pointcloud_queue_size_, &SemTsdfServer::integratePointcloud, this);
    }
    void SemTsdfServer::integratePointcloud(const sensor_msgs::PointCloud2::Ptr &pointcloud_msg_in)
    {
        if (pointcloud_msg_in->header.stamp - last_msg_time_ptcloud_ > min_time_between_msgs_)
        {
            last_msg_time_ptcloud_ = pointcloud_msg_in->header.stamp;
            pointcloud_queue_.push(pointcloud_msg_in);
        }

        Transformation T_G_C;
        sensor_msgs::PointCloud2::Ptr pointcloud_msg;
        bool processed_any = false;

        while (getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg, &T_G_C))
        {
            constexpr bool is_freespace_pointcloud = false;
            processPointCloudMessageAndInsert(pointcloud_msg, T_G_C, is_freespace_pointcloud);
            processed_any = true;
        }
        if (!processed_any)
        {
            return;
        }
        if (publish_pointclouds_on_update_)
        {
            publishPointclouds();
        }
    }

    // Check if we can get the next message from queue
    bool SemTsdfServer::getNextPointcloudFromQueue(
        std::queue<sensor_msgs::PointCloud2::Ptr> *queue,
        sensor_msgs::PointCloud2::Ptr *pointcloud_msg, Transformation *T_G_C)
    {
        const size_t kMaxQueueSize = 10;
        if (queue->empty())
        {
            return false;
        }
        *pointcloud_msg = queue->front();

        if (transformer_.lookupTransform((*pointcloud_msg)->header.frame_id, world_frame_id_, (*pointcloud_msg)->header.stamp, T_G_C))
        {
            queue->pop();
            return true;
        }
        else
        {
            if (queue->size() >= kMaxQueueSize)
            {
                ROS_ERROR_THROTTLE(60,
                                   "Input pointcloud queue getting too long! Dropping "
                                   "some pointclouds. Either unable to look up transform "
                                   "timestamps or the processing is taking too long.");
            }
        }
    }
    // 
    void SemTsdfServer::processPointCloudMessageAndInsert(
        const sensor_msgs::PointCloud2::Ptr &pointcloud_msg,
        const Transformation &T_G_C, const bool is_freespace_pointcloud)
    {
        Pointcloud points_C;
        Colors colors;
        pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
        pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
        Transformation T_G_C_refined = T_G_C;
        // TODO: Enable ICP
        // integratePointcloud(T_G_C_refined, points_C, colors, is_freespace_pointcloud);
        // tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
        //     T_G_C.getPosition(), max_block_distance_from_body_);
        // mesh_layer_->clearDistantMesh(T_G_C.getPosition(),
        //                               max_block_distance_from_body_);
        }
    void SemTsdfServer::publishPointclouds()
    {
    }
}
