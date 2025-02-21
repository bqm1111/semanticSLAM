#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import os
from sensor_msgs.msg import PointCloud2
from color_pcl_generator import PointType, ColorPclGenerator
from diffusionMMS.engine import get_model
from diffusionMMS.utils.helper import convert_depth_to_three_channel_img, get_class_colors
import message_filters
import time
from omegaconf import OmegaConf
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import copy
import torch
import cv2


def convert_to_pointcloud2(points, stamp, frame_id="camera_link"):
    header = rospy.Header()
    header.stamp = stamp
    header.frame_id = frame_id  # Set the frame ID

    # Define PointField structure
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # Create PointCloud2 message
    cloud_msg = pc2.create_cloud(header, fields, points)
    return cloud_msg


def visualize(pred, num_classes=40):
    # Get color corresponding to each classes
    colors = np.array(get_class_colors(num_classes + 1))

    # Convert data to unit8 numpy type on cpu
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    colored_pred = np.zeros_like(pred_arr)
    colored_pred = np.stack((colored_pred,) * 3, axis=-1)
    colored_pred[:] = colors[pred_arr[:]]

    return colored_pred


def setup_model(cfg_file, device):
    config = OmegaConf.load(cfg_file)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = os.path.join(
        "output_dir/",
        config.experiment_dataset,
        config.experiment_name,
        "checkpoint-" + str(93) + ".pth",
    )

    model.load_state_dict(torch.load(checkpoint_path)["model"])
    model = model.to(device)
    return model


def preprocess(input_rgb, input_depth):
    depth = copy.copy(input_depth)
    rgb = copy.copy(input_rgb)
    depth[np.isnan(depth)] = 0  # Replace NaN with 0
    depth[np.isinf(depth)] = 0  # Replace Inf with 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb / 255.0 - mean) / std

    depth = convert_depth_to_three_channel_img(depth) / 255.0

    rgb = rgb.transpose(2, 0, 1)
    depth = depth.transpose(2, 0, 1)

    rgb = torch.from_numpy(rgb).unsqueeze(0).cuda().float()
    depth = torch.from_numpy(depth).unsqueeze(0).cuda().float()

    output = {"rgb": rgb, "depth": depth}

    return output


def predict(model, rgb, depth):
    data = preprocess(rgb, depth)
    with torch.no_grad():
        score = model.sampling(data["rgb"], data["depth"])
    pred = score.argmax(1)
    return pred


class SemanticCloud:
    def __init__(self, seg_cfg_file):
        # Get point type
        # Get image size
        self.img_width, self.img_height = rospy.get_param(
            "/camera/width"
        ), rospy.get_param("/camera/height")
        # Set up semantic segmentation model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = setup_model(seg_cfg_file, device)
        self.model.eval()

        # Set up ROS
        print("Setting up ROS...")
        # CvBridge to transform ROS Image message to OpenCV image
        self.bridge = CvBridge()
        # Semantic image publisher
        self.sem_img_pub = rospy.Publisher(
            "/semantic_pcl/semantic_image", Image, queue_size=1
        )
        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        # Point cloud frame id
        frame_id = rospy.get_param("/semantic_pcl/frame_id")
        # Camera intrinsic matrix
        self.fx = rospy.get_param("/camera/fx")
        self.fy = rospy.get_param("/camera/fy")
        self.cx = rospy.get_param("/camera/cx")
        self.cy = rospy.get_param("/camera/cy")
        self.pcl_pub = rospy.Publisher(
            "/semantic_pcl/semantic_pcl", PointCloud2, queue_size=1
        )

        self.color_sub = message_filters.Subscriber(
            rospy.get_param("/semantic_pcl/color_image_topic"),
            Image,
            queue_size=1,
            buff_size=30 * 480 * 640,
        )
        self.depth_sub = message_filters.Subscriber(
            rospy.get_param("/semantic_pcl/depth_image_topic"),
            Image,
            queue_size=1,
            buff_size=40 * 480 * 640,
        )  # increase buffer size to avoid delay (despite queue_size = 1)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=1, slop=0.3
        )  # Take in one color image and one depth image with a limite time gap between message time stamps
        self.ts.registerCallback(self.color_depth_callback)
        print("Ready.")

    def color_depth_callback(self, color_img_ros, depth_img_ros):
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_img_ros, desired_encoding="32FC1"
            )
        except CvBridgeError as e:
            print(e)

        semantic_pred = predict(self.model, color_img, depth_img)
        semantic_color = visualize(semantic_pred, num_classes=40)
        if self.sem_img_pub.get_num_connections() > 0:
            if self.point_type is PointType.SEMANTICS_MAX:
                semantic_color_msg = self.bridge.cv2_to_imgmsg(
                    semantic_color, encoding="bgr8"
                )
            else:
                semantic_color_msg = self.bridge.cv2_to_imgmsg(
                    self.semantic_colors[0], encoding="bgr8"
                )
            self.sem_img_pub.publish(semantic_color_msg)

        x, y = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
        x, y, d = x.flatten(), y.flatten(), depth_img.flatten()
        valid_mask = ~np.isnan(d)

        x, y, d = x[valid_mask], y[valid_mask], d[valid_mask]
        x = (x - self.cx) * d / self.fx
        y = (y - self.cy) * d / self.fy
        z = d
        
        points = np.stack((x, y, z), axis=-1)
        cloud_ros = convert_to_pointcloud2(points, stamp=color_img_ros.header.stamp)

        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)


def main(args):
    rospy.init_node("semantic_cloud", anonymous=True)
    cfg_file = "semantic_cloud/src/diffusionMMS/config/nyuv2/standard/ddp_dual_dat_s_mmcv_epoch_100.yaml"

    SemanticCloud(seg_cfg_file=cfg_file)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
