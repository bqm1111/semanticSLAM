#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from diffusionMMS.engine import get_model
from diffusionMMS.utils.helper import (
    convert_depth_to_three_channel_img,
    get_class_colors,
)
import cv2
import message_filters
import time
from omegaconf import OmegaConf
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import copy
import torch
import struct
import tf
import tf2_ros
import tf.transformations as tf_trans
from dataclasses import dataclass
from typing import Optional, Tuple, List
from helper import (convert_to_pointcloud2, depth_to_color_frame, erode_segmentation_mask, filter_flying_depth_data, advanced_filter_depth_with_mask)
import numpy as np
import cv2

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    enable_semantic: bool
    selected_semantic: List[str]
    process_semantic_freq: int
    depth_scale: float

class SemanticSegmentationModel():
    """Wrapper for semantic segmentation model."""
    def __init__(self, cfg_file: str, ckpt_path:str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._setup_model(cfg_file, self.device, ckpt_path)
        self.model.eval()
    
    def _setup_model(self, cfg_file, device, ckpt_path):
        config = OmegaConf.load(cfg_file)
        model = get_model(config.model.name, eval=True, **config.model.params)
        checkpoint_path = ckpt_path

        model.load_state_dict(torch.load(checkpoint_path)["model"])
        model = model.to(device)
        return model

    @staticmethod
    def preprocess(input_rgb, input_depth):
        depth = copy.copy(input_depth)
        rgb = copy.copy(input_rgb)
        depth[np.isnan(depth)] = 0  
        depth[np.isinf(depth)] = 0  

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

    @staticmethod
    def colorized_prediction(pred, num_classes=40):
        """Colorized semantic segmentation masks and erode boundaries of object masks"""
        # Get color corresponding to each classes
        colors = np.array(get_class_colors(num_classes + 1))

        # Convert data to unit8 numpy type on cpu
        pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        trimmed_mask, valid_mask = erode_segmentation_mask(pred_arr, kernel_size=5, iterations=4)
        colored_pred = np.zeros_like(pred_arr)
        colored_pred = np.stack((colored_pred,) * 3, axis=-1)
        colored_pred[:] = colors[trimmed_mask[:]]

        return colored_pred, np.array(trimmed_mask), valid_mask

    def predict(self, rgb, depth):
        data = self.preprocess(rgb, depth)
        with torch.no_grad():
            score = self.model.sampling(data["rgb"], data["depth"])
        pred = score.argmax(1)
        return pred


class TransformManager:
    """Manages coordinate frame transformations."""
    
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(3)  # Wait for TF buffer to fill
    
    def get_transform_matrix(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        """Get 4x4 transformation matrix between frames."""
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0)
            )
            
            # Extract translation and rotation
            translation = trans.transform.translation
            t = np.array([translation.x, translation.y, translation.z])
            
            rotation = trans.transform.rotation
            q = [rotation.x, rotation.y, rotation.z, rotation.w]
            R = tf_trans.quaternion_matrix(q)[:3, :3]
            
            # Construct transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            return transform_matrix
            
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return None

class SemanticCloud:
    SEMANTIC_CLASSES = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
        "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
        "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
        "books", "refridgerator", "television", "paper", "towel", "shower curtain",
        "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
        "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop"
    ]

    def __init__(self, seg_cfg_file):
        self.bridge = CvBridge()
        self.config = self._load_config()
        self.transform_manager = TransformManager()

        # Set up segmentation model
        self.seg_model = None
        if self.config.enable_semantic:
            ckpt_path = rospy.get_param("semseg_model_ckpt")
            if not os.path.exists(seg_cfg_file):
                rospy.logerr(f"Semseg model config file not found: {seg_cfg_file}")
            else:
                rospy.loginfo(f"Using Semseg model config file: {seg_cfg_file}")                
            if not os.path.exists(ckpt_path):
                rospy.logerr(f"pretrained semseg model file not found: {ckpt_path}")
            else:
                rospy.loginfo(f"Using pretrained model file: {ckpt_path}")
            self.seg_model = SemanticSegmentationModel(seg_cfg_file, ckpt_path)

        self._setup_semantic_class_filtering()        
        self._setup_ros()
        self._setup_transforms()
        
        # State variables
        self.counter = 0
        self.num_aggregated_observations = 0
        self.latest_pose = None
        rospy.loginfo("Semantic Point Cloud Node initialized successfully")

    def _load_config(self):
        """Load configuration from ROS parameters"""
        return ProcessingConfig(enable_semantic=rospy.get_param("enable_semantic"),
                                selected_semantic=rospy.get_param("selected_semantic"),
                                process_semantic_freq=rospy.get_param("process_semantic_freq"),
                                depth_scale=rospy.get_param("depth_scale", 0.001))

    def _setup_semantic_class_filtering(self):
        """Setup semantic class filtering."""
        if "all" not in self.config.selected_semantic:
            index_map = {val: idx for idx, val in enumerate(self.SEMANTIC_CLASSES)}
            self.semantic_index_map = np.array([index_map[val] for val in self.config.selected_semantic])
        else:
            self.semantic_index_map = None
    
    def _setup_transforms(self):
        """Setup coordinate frame transforms."""
        color_frame = rospy.get_param("d455_color_optical_frame")
        depth_frame = rospy.get_param("d455_depth_optical_frame")
        
        self.depth_to_color_transform = self.transform_manager.get_transform_matrix(color_frame, depth_frame
        )
        self.color_to_depth_transform = self.transform_manager.get_transform_matrix(depth_frame, color_frame)
        
        if self.depth_to_color_transform is None or self.color_to_depth_transform is None:
            rospy.logfatal("Failed to get required transforms")
            sys.exit(1)
            
    def _setup_ros(self):
        # Set up ROS
        print("Setting up ROS...")
        # Frame id
        self.frame_id = rospy.get_param("semantic_pcl_frame_id")

        # Publishers
        self.sem_img_pub = rospy.Publisher("/semantic_pcl/semantic_image", Image, queue_size=1)
        self.pcl_pub = rospy.Publisher("/semantic_pcl/semantic_pcl", PointCloud2, queue_size=1)

        # Subscribers
        camera_pose_topic = rospy.get_param("camera_pose_topic")
        color_image_topic = rospy.get_param("color_image_topic")
        depth_image_topic = rospy.get_param("depth_image_topic")

        color_cam_info_topic = rospy.get_param("color_cam_info_topic")
        depth_cam_info_topic = rospy.get_param("depth_cam_info_topic")


        self.color_sub = message_filters.Subscriber( color_image_topic,Image,queue_size=60,buff_size=30 * 480 * 848)
        self.depth_sub = message_filters.Subscriber(depth_image_topic,Image,queue_size=60,buff_size=40 * 480 * 848,)  # increase buffer size to avoid delay (despite queue_size = 1)
        self.color_cam_info_sub = message_filters.Subscriber(color_cam_info_topic, CameraInfo)
        self.depth_cam_info_sub = message_filters.Subscriber(depth_cam_info_topic, CameraInfo)
        self.camera_pose_sub = rospy.Subscriber(camera_pose_topic, Odometry, self._pose_callback)
        
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                self.color_sub,
                self.depth_sub,
                self.color_cam_info_sub,
                self.depth_cam_info_sub,
            ],
            queue_size=10,
            slop=0.05,
        )
        
        self.ts.registerCallback(self._sensor_callback)

    def _pose_callback(self, msg):
        """Callback function to store the latest camera pose."""
        self.latest_pose = msg

    def transform_point_cloud(self, points, T):
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        transformed_points = np.hstack((points, ones))  # Avoid unnecessary transposes
        transformed_points = np.matmul(transformed_points, T.T)[:, :3]
        return transformed_points
                
    def _sensor_callback(
        self, color_msg:Image, depth_msg:Image, color_info:CameraInfo, depth_info:CameraInfo
    ):
        """Main sensor data processing callback"""
        # Convert ros Image message to numpy array
        self.counter += 1
        if self.counter % self.config.process_semantic_freq == 0 and self.latest_pose is not None:
            try:
                self._process_sensor_data(color_msg, depth_msg, color_info, depth_info)
                self.num_aggregated_observations +=1
                rospy.loginfo(f"Processed observation #{self.num_aggregated_observations}")
            except Exception as e:
                rospy.logerr(f"Error processing sensor data: {e}")

    def _process_sensor_data(self, color_msg: Image, depth_msg: Image,
                           color_info: CameraInfo, depth_info: CameraInfo):
        """Process RGB-D sensor data to generate semantic point cloud."""
        # Convert ROS messages to OpenCV format
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="16UC1"
            )
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # Get camera intrinsics
        K_color = np.array(color_info.K).reshape(3, 3)
        K_depth = np.array(depth_info.K).reshape(3, 3)
        # Align depth image to color image frame
        aligned_depth, fx, fy, cx, cy, img_height, img_width = depth_to_color_frame(
            depth_img,
            self.config.depth_scale,
            K_depth,
            K_color,
            self.depth_to_color_transform,
        )
        # Generate semantic segmentation mask
        if self.config.enable_semantic:
            semantic_pred = self.seg_model.predict(color_img, aligned_depth)
            semantic_color, pred_arr, eroded_valid_mask = self.seg_model.colorized_prediction(semantic_pred, num_classes=len(self.SEMANTIC_CLASSES))
            semantic_color_msg = self.bridge.cv2_to_imgmsg(
                semantic_color, encoding="bgr8"
            )
            self.sem_img_pub.publish(semantic_color_msg)
            colors = cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB)
        else:          
            colors = np.ones_like(color_img) * 100
        colors = colors.reshape(-1, 3)
        
        # Filter depth inaccuracies
        depth_meters = depth_img.astype(np.float32) * self.config.depth_scale
        x, y = np.meshgrid(
            np.arange(img_width), np.arange(img_height), indexing="xy"
        )
        # Specifically filter depth pixel labeled as 'wall' because this class has the most depth inaccuracies
        depth_wall = np.where(pred_arr ==0, depth_meters, 0)
        flying_wall_mask = advanced_filter_depth_with_mask(np.array(depth_wall), gradient_thresh=0.1, min_valid_neighbors=3)
        flying_depth_valid_mask = filter_flying_depth_data(depth_meters, gradient_thresh=0.1)
        flying_depth_valid_mask = flying_depth_valid_mask.ravel()
        eroded_valid_mask = eroded_valid_mask.ravel()
        flying_wall_mask = ~flying_wall_mask.ravel()
        
        if self.config.enable_semantic:
            x, y, d, pred_arr = x.ravel(), y.ravel(), aligned_depth.ravel(), pred_arr.ravel()
        else:
            x, y, d, pred_arr = x.ravel(), y.ravel(), depth_meters.ravel(), pred_arr.ravel()
        
        # Create the filtered depth mask 
        filtered_depth_mask = np.isfinite(d) & (d != 0) & eroded_valid_mask & flying_depth_valid_mask & flying_wall_mask

        if self.semantic_index_map is not None:
            filtered_depth_mask = filtered_depth_mask & np.isin(pred_arr, self.semantic_index_map)

        # Generate semantic pointcloud
        x, y, d, colors = (
            x[filtered_depth_mask],
            y[filtered_depth_mask],
            d[filtered_depth_mask],
            colors[filtered_depth_mask],
        )

        x = (x - cx) * d / fx
        y = (y - cy) * d / fy
        z = d

        points = np.column_stack((x, y, z))
        points = self.transform_point_cloud(points, self.color_to_depth_transform)
        cloud_data = np.empty((points.shape[0], 4), dtype=np.float32)
        cloud_data[:, :3] = points

        # Pack RGB and store in the last column
        cloud_data[:, 3] = (
            (colors[:, 0].astype(np.uint32) << 16)
            | (colors[:, 1].astype(np.uint32) << 8)
            | colors[:, 2].astype(np.uint32)
        ).view(np.float32)
        cloud_ros = convert_to_pointcloud2(
            cloud_data, stamp=color_msg.header.stamp, frame_id=self.frame_id
        )
        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)
    
def main(args):
    rospy.init_node("semantic_cloud", anonymous=True)
    cfg_file = rospy.get_param("semseg_model_cfg")
    SemanticCloud(seg_cfg_file=cfg_file)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
