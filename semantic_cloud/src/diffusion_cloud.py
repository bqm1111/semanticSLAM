#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

import numpy as np
import cv2
from numba import njit, prange

SEMANTIC_CLASSES_NAMES = ["wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "blinds",
            "desk",
            "shelves",
            "curtain",
            "dresser",
            "pillow",
            "mirror",
            "floor mat",
            "clothes",
            "ceiling",
            "books",
            "refridgerator",
            "television",
            "paper",
            "towel",
            "shower curtain",
            "box",
            "whiteboard",
            "person",
            "night stand",
            "toilet",
            "sink",
            "lamp",
            "bathtub",
            "bag",
            "otherstructure",
            "otherfurniture",
            "otherprop"]
@njit
def get_unique_labels(mask):
    return np.unique(mask)

def erode_segmentation_mask(
    mask: np.ndarray, kernel_size: int = 3, iterations: int = 1
):
    """
    Erodes each object mask in the semantic segmentation mask.

    Args:
        mask (np.ndarray): HxW array where each pixel is a class label (e.g., 0 for background, 1, 2, ... for objects).
        kernel_size (int): Size of the erosion kernel.
        iterations (int): Number of erosion iterations.

    Returns:
        np.ndarray: Boolean mask of valid pixels (True = keep, False = remove).
    """
    unique_labels = get_unique_labels(mask)
    valid_mask = np.zeros_like(mask, dtype=bool)
    trimmed_mask = np.zeros_like(mask, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for label in unique_labels:
        binary_mask = (mask == label).astype(np.uint8)
        if label == 0:
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=5)
        else:
            eroded_mask = cv2.erode(binary_mask, kernel, iterations=2)
        valid_mask |= eroded_mask.astype(bool)
        trimmed_mask[eroded_mask == 1] = label

    return trimmed_mask, valid_mask

@njit
def filter_flying_depth_data(depth_image, gradient_thresh=0.1):
    h, w = depth_image.shape
    filtered = np.zeros_like(depth_image)
    mask = np.zeros((h, w), dtype=np.bool_)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = depth_image[y, x]
            if center == 0:
                continue

            up    = depth_image[y - 1, x]
            down  = depth_image[y + 1, x]
            left  = depth_image[y, x - 1]
            right = depth_image[y, x + 1]

            if up == 0 or down == 0 or left == 0 or right == 0:
                continue

            if (abs(center - up) < gradient_thresh and
                abs(center - down) < gradient_thresh and
                abs(center - left) < gradient_thresh and
                abs(center - right) < gradient_thresh):
                filtered[y, x] = center
                mask[y, x] = True

    return mask

@njit(fastmath=True, parallel=True)
def depth_to_color_frame(
    depth_image_raw, depth_scale, depth_intrinsics, color_intrinsics, transform_matrix
):
    """
    Convert a depth image from the depth optical frame to the color optical frame.
    Only keeps pixels inside the bounding box of valid depth projections.
    """

    depth_meters = depth_image_raw.astype(np.float32) * depth_scale
    h, w = depth_meters.shape

    fx_d, fy_d, cx_d, cy_d = (
        depth_intrinsics[0, 0],
        depth_intrinsics[1, 1],
        depth_intrinsics[0, 2],
        depth_intrinsics[1, 2],
    )
    fx_c, fy_c, cx_c, cy_c = (
        color_intrinsics[0, 0],
        color_intrinsics[1, 1],
        color_intrinsics[0, 2],
        color_intrinsics[1, 2],
    )

    inv_fx_d, inv_fy_d = 1.0 / fx_d, 1.0 / fy_d

    depth_color_aligned = np.zeros((h, w), dtype=np.float32)  # Initialize to 0

    r11, r12, r13, t1 = (
        transform_matrix[0, 0],
        transform_matrix[0, 1],
        transform_matrix[0, 2],
        transform_matrix[0, 3],
    )
    r21, r22, r23, t2 = (
        transform_matrix[1, 0],
        transform_matrix[1, 1],
        transform_matrix[1, 2],
        transform_matrix[1, 3],
    )
    r31, r32, r33, t3 = (
        transform_matrix[2, 0],
        transform_matrix[2, 1],
        transform_matrix[2, 2],
        transform_matrix[2, 3],
    )
    min_u, min_v = w, h
    max_u, max_v = 0, 0
    # 
    # Store transformed coordinates
    transformed_coords = np.full((h, w, 3), np.nan, dtype=np.float32)  # Stores X, Y, Z
    projected_coords = np.full((h, w, 2), -1, dtype=np.int32)  # Stores u, v

    # Pass 1: Compute bounding box and store transformed coordinates
    for y in prange(h):  # Outer loop parallelized
        for x in range(w):  # Inner loop sequential
            z = depth_meters[y, x]
            if z <= 0:
                continue  # Ignore invalid depth

            xd = (x - cx_d) * inv_fx_d * z
            yd = (y - cy_d) * inv_fy_d * z

            # Transform to color frame
            X = r11 * xd + r12 * yd + r13 * z + t1
            Y = r21 * xd + r22 * yd + r23 * z + t2
            Z = r31 * xd + r32 * yd + r33 * z + t3

            if Z <= 0:
                continue  # Ignore invalid projections

            # Project to color image plane
            u = int(X * fx_c / Z + cx_c)
            v = int(Y * fy_c / Z + cy_c)

            if 0 <= u < w and 0 <= v < h:  # Valid projection
                transformed_coords[y, x, 0] = X
                transformed_coords[y, x, 1] = Y
                transformed_coords[y, x, 2] = Z
                projected_coords[y, x, 0] = u
                projected_coords[y, x, 1] = v

                min_u = min(min_u, u)
                max_u = max(max_u, u)
                min_v = min(min_v, v)
                max_v = max(max_v, v)

    # Pass 2: Map depth to color frame using stored values
    for y in prange(h):  # Again, only outer loop parallelized
        for x in range(w):  # Inner loop sequential
            if np.isnan(transformed_coords[y, x, 0]):  # Skip invalid points
                continue

            X, Y, Z = transformed_coords[y, x]
            u, v = projected_coords[y, x]

            if min_u <= u <= max_u and min_v <= v <= max_v:  # Inside bounding box
                depth_color_aligned[v, u] = Z

    return depth_color_aligned, fx_d, fy_d, cx_d, cy_d, h, w


def get_transform(target_frame, source_frame):
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(3)

    try:
        trans = tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0)
        )

        # Extract translation
        translation = trans.transform.translation
        t = np.array([translation.x, translation.y, translation.z])

        # Extract rotation (quaternion)
        rotation = trans.transform.rotation
        q = [rotation.x, rotation.y, rotation.z, rotation.w]

        # Convert quaternion to rotation matrix
        R = tf_trans.quaternion_matrix(q)[:3, :3]  # Extract 3x3 rotation matrix

        # Construct 4x4 transformation matrix
        depth_to_color_matrix = np.eye(4)
        depth_to_color_matrix[:3, :3] = R
        depth_to_color_matrix[:3, 3] = t

        return depth_to_color_matrix

    except tf2_ros.LookupException:
        rospy.logerr(f"Transform between {source_frame} and {target_frame} not found.")
    except tf2_ros.ExtrapolationException:
        rospy.logerr("TF extrapolation error.")
    except rospy.ROSInterruptException:
        pass

# def convert_to_pointcloud2(points, stamp, frame_id="camera_rgb_optical_frame"):
#     # Create header
#     header = rospy.Header(stamp=stamp, frame_id=frame_id)

#     # Ensure points is a NumPy array with dtype float32
#     points = np.asarray(points, dtype=np.float32)
    
#     # Create RGBA field efficiently
#     rgba = np.zeros(points.shape[0], dtype=np.uint32)
#     rgba[:] = 255 << 24 | (points[:, 3].astype(np.uint8))  # Pack RGBA with alpha 255
#     rgba = rgba.view(np.float32)  # Convert to float32 for ROS format
    
#     # Concatenate x, y, z, and rgba
#     points_with_rgba = np.column_stack((points[:, :3], rgba))

#     # Create PointCloud2 message
#     cloud_msg = PointCloud2()
#     cloud_msg.header = header
#     cloud_msg.height = 1  # Unorganized point cloud
#     cloud_msg.width = points_with_rgba.shape[0]
#     cloud_msg.is_dense = True  # No NaN values
#     cloud_msg.is_bigendian = False

#     # Define fields
#     cloud_msg.fields = [
#         PointField("x", 0, PointField.FLOAT32, 1),
#         PointField("y", 4, PointField.FLOAT32, 1),
#         PointField("z", 8, PointField.FLOAT32, 1),
#         PointField("rgba", 12, PointField.FLOAT32, 1),
#     ]

#     cloud_msg.point_step = 16  # 4 floats (x, y, z, rgba) * 4 bytes each
#     cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

#     # Convert NumPy array to binary format for ROS
#     cloud_msg.data = points_with_rgba.tobytes()
#     return cloud_msg

def convert_to_pointcloud2(points, stamp, frame_id="camera_rgb_optical_frame"):
    # Create header
    header = rospy.Header(stamp=stamp, frame_id=frame_id)

    # Ensure points is a NumPy array with dtype float32
    points = np.asarray(points, dtype=np.float32)

    # Faster alternative: Directly create a PointCloud2 message
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1  # Unorganized point cloud
    cloud_msg.width = points.shape[0]
    cloud_msg.is_dense = True  # No NaN values
    cloud_msg.is_bigendian = False

    # Define fields
    cloud_msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.FLOAT32, 1),
    ]

    cloud_msg.point_step = 16  # 4 floats (x, y, z, rgb) * 4 bytes each
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

    # Convert NumPy array to binary format for ROS
    cloud_msg.data = points.tobytes()  # Efficient binary serialization
    return cloud_msg


def colorized_prediction(pred, num_classes=40):
    # Get color corresponding to each classes
    colors = np.array(get_class_colors(num_classes + 1))

    # Convert data to unit8 numpy type on cpu
    pred_arr = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    trimmed_mask, valid_mask = erode_segmentation_mask(pred_arr, kernel_size=5, iterations=4)
    colored_pred = np.zeros_like(pred_arr)
    colored_pred = np.stack((colored_pred,) * 3, axis=-1)
    colored_pred[:] = colors[trimmed_mask[:]]

    return colored_pred, np.array(trimmed_mask), valid_mask


def setup_model(cfg_file, device, ckpt_path):
    config = OmegaConf.load(cfg_file)
    model = get_model(config.model.name, eval=True, **config.model.params)
    checkpoint_path = ckpt_path

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

@njit
def advanced_filter_depth_with_mask(depth_image, gradient_thresh=0.1, min_valid_neighbors=10
):
    h, w = depth_image.shape
    filtered = np.zeros_like(depth_image)
    mask = np.zeros((h, w), dtype=np.bool_)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            center = depth_image[y, x]
            if center == 0:
                continue

            # Neighbor values
            neighbors = [
                depth_image[y - 1, x],
                depth_image[y + 1, x],
                depth_image[y, x - 1],
                depth_image[y, x + 1]
            ]

            # Count how many neighbors are valid (non-zero)
            valid_count = 0
            for v in neighbors:
                if v != 0:
                    valid_count += 1

            if valid_count < min_valid_neighbors:
                continue  # Likely a bad region

            # Now check gradients
            if (abs(center - neighbors[0]) < gradient_thresh and
                abs(center - neighbors[1]) < gradient_thresh and
                abs(center - neighbors[2]) < gradient_thresh and
                abs(center - neighbors[3]) < gradient_thresh):
                filtered[y, x] = center
                mask[y, x] = False
            else:
                mask[y, x] = True 

    return mask

class SemanticCloud:
    def __init__(self, seg_cfg_file):
        # Set up semantic segmentation model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.enable_semantic = rospy.get_param("enable_semantic")
        self.selected_semantic = rospy.get_param("selected_semantic")
        if "all" not in self.selected_semantic:
            index_map = {val: idx for idx, val in enumerate(SEMANTIC_CLASSES_NAMES)}
            self.semantic_index_map = np.array([index_map[val] for val in self.selected_semantic])
        else:
            self.semantic_index_map = None

        if self.enable_semantic:
            ckpt_path = rospy.get_param("semseg_model_ckpt")
            self.model = setup_model(seg_cfg_file, device, ckpt_path)
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
        self.frame_id = rospy.get_param("semantic_pcl_frame_id")

        self.pcl_pub = rospy.Publisher(
            "/semantic_pcl/semantic_pcl", PointCloud2, queue_size=1
        )
        camera_pose_topic = rospy.get_param("camera_pose_topic")
        color_image_topic = rospy.get_param("color_image_topic")
        depth_image_topic = rospy.get_param("depth_image_topic")

        color_cam_info_topic = rospy.get_param("color_cam_info_topic")
        depth_cam_info_topic = rospy.get_param("depth_cam_info_topic")

        tf_color_frame = rospy.get_param("d455_color_optical_frame")
        tf_depth_frame = rospy.get_param("d455_depth_optical_frame")
        self.process_sematic_freq = rospy.get_param("process_sematic_freq")
        self.depth_to_color_transform = get_transform(
            target_frame=tf_color_frame, source_frame=tf_depth_frame
        )
        self.color_to_depth_transform = get_transform(
            target_frame=tf_depth_frame, source_frame=tf_color_frame
        )
        self.optical_normal_convention_depth_transform = get_transform(
            target_frame="d455_depth_frame", source_frame="d455_depth_optical_frame"
        )

        self.depth_scale = rospy.get_param("depth_scale", 0.001)

        self.color_sub = message_filters.Subscriber(
            color_image_topic,
            Image,
            queue_size=60,
            buff_size=30 * 480 * 848,
        )
        self.depth_sub = message_filters.Subscriber(
            depth_image_topic,
            Image,
            queue_size=60,
            buff_size=40 * 480 * 848,
        )  # increase buffer size to avoid delay (despite queue_size = 1)
        self.color_cam_info_sub = message_filters.Subscriber(
            color_cam_info_topic, CameraInfo
        )
        self.depth_cam_info_sub = message_filters.Subscriber(
            depth_cam_info_topic, CameraInfo
        )

        self.camera_pose = rospy.Subscriber(
            camera_pose_topic, Odometry, self.pose_callback
        )
        
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
        self.ts.registerCallback(self.color_depth_callback)
        self.bridge = CvBridge()
        self.counter = 0
        self.num_aggregated_observations = 0
        self.latest_pose = None
        print("Ready.")

    def pose_callback(self, msg):
        """Callback function to store the latest camera pose."""
        self.latest_pose = msg

    def pose_to_matrix(self, odom):
        """Convert Odometry to a 4x4 transformation matrix."""
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        trans = np.array([position.x, position.y, position.z])
        rot = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        # Convert quaternion to rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]

        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = trans
        return T

    def transform_point_cloud(self, points, T):
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        transformed_points = np.hstack((points, ones))  # Avoid unnecessary transposes
        transformed_points = np.matmul(transformed_points, T.T)[:, :3]
        return transformed_points
    
    def color_depth_callback(
        self, color_msg, depth_msg, color_caminfo_msg, depth_caminfo_msg
    ):
        # Convert ros Image message to numpy array
        self.counter += 1
        if (
            self.counter % self.process_sematic_freq == 0
            and self.latest_pose is not None
        ):
            try:
                color_img = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")
                depth_img = self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding="16UC1"
                )
            except CvBridgeError as e:
                print(e)
            self.num_aggregated_observations += 1
            print("============> Number of aggregated observations: ", self.num_aggregated_observations)
            K_color = np.array(color_caminfo_msg.K).reshape(3, 3)
            K_depth = np.array(depth_caminfo_msg.K).reshape(3, 3)

            aligned_depth, fx, fy, cx, cy, img_height, img_width = depth_to_color_frame(
                depth_img,
                self.depth_scale,
                K_depth,
                K_color,
                self.depth_to_color_transform,
            )

            if self.enable_semantic:
                semantic_pred = predict(self.model, color_img, aligned_depth)
                semantic_color, pred_arr, eroded_valid_mask = colorized_prediction(semantic_pred, num_classes=40)
                semantic_color_msg = self.bridge.cv2_to_imgmsg(
                    semantic_color, encoding="bgr8"
                )
                self.sem_img_pub.publish(semantic_color_msg)
                colors = cv2.cvtColor(semantic_color, cv2.COLOR_BGR2RGB)
            else:          
                colors = np.ones_like(color_img) * 100
            colors = colors.reshape(-1, 3)
            depth_meters = depth_img.astype(np.float32) * self.depth_scale
            x, y = np.meshgrid(
                np.arange(img_width), np.arange(img_height), indexing="xy"
            )
            depth_wall = np.where(pred_arr ==0, depth_meters, 0)
            flying_wall_mask = advanced_filter_depth_with_mask(np.array(depth_wall), gradient_thresh=0.1, min_valid_neighbors=3)
            flying_depth_valid_mask = filter_flying_depth_data(depth_meters, gradient_thresh=0.1)
            
            flying_depth_valid_mask = flying_depth_valid_mask.ravel()
            eroded_valid_mask = eroded_valid_mask.ravel()
            flying_wall_mask = ~flying_wall_mask.ravel()
            if self.enable_semantic:
                x, y, d, pred_arr = x.ravel(), y.ravel(), aligned_depth.ravel(), pred_arr.ravel()
            else:
                x, y, d, pred_arr = x.ravel(), y.ravel(), depth_meters.ravel(), pred_arr.ravel()
            
            valid_mask = np.isfinite(d) & (d != 0) & eroded_valid_mask & flying_depth_valid_mask & flying_wall_mask

            if self.semantic_index_map is not None:
                valid_mask = valid_mask & np.isin(pred_arr, self.semantic_index_map)

            x, y, d, colors = (
                x[valid_mask],
                y[valid_mask],
                d[valid_mask],
                colors[valid_mask],
            )

            x = (x - cx) * d / fx
            y = (y - cy) * d / fy
            z = d

            points = np.column_stack((x, y, z))
            points = self.transform_point_cloud(points, self.color_to_depth_transform)
            # Directly allocate memory for the final cloud data (avoids copies)
            cloud_data = np.empty((points.shape[0], 4), dtype=np.float32)

            # Fill XYZ directly
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
