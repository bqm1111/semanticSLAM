#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

from sensor_msgs.msg import PointCloud2
from color_pcl_generator import PointType, ColorPclGenerator
import message_filters
import time
from omegaconf import OmegaConf
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from skimage.transform import resize
import cv2

import torch
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict


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


def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format (rgb)
    \param N (int) number of classes
    \param normalized (bool) whether colors are normalized (float 0-1)
    \return (Nx3 numpy array) a color map
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap / 255.0 if normalized else cmap
    return cmap


def decode_segmap(temp, n_classes, cmap):
    """
    Given an image of class predictions, produce an bgr8 image with class colors
    \param temp (2d numpy int array) input image with semantic classes (as integer)
    \param n_classes (int) number of classes
    \cmap (Nx3 numpy array) input color map
    \return (numpy array bgr8) the decoded image with class colors
    """
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = cmap[l, 0]
        g[temp == l] = cmap[l, 1]
        b[temp == l] = cmap[l, 2]
    bgr = np.zeros((temp.shape[0], temp.shape[1], 3))
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr.astype(np.uint8)


class SemanticCloud:
    """
    Class for ros node to take in a color image (bgr) and do semantic segmantation on it to produce an image with semantic class colors (chair, desk etc.)
    Then produce point cloud based on depth information
    CNN: PSPNet (https://arxiv.org/abs/1612.01105) (with resnet50) pretrained on ADE20K, fine tuned on SUNRGBD or not
    """

    def __init__(self):
        """
        Constructor
        \param gen_pcl (bool) whether generate point cloud, if set to true the node will subscribe to depth image
        """
        # Get point type
        point_type = rospy.get_param("/semantic_pcl/point_type")
        if point_type == 0:
            self.point_type = PointType.COLOR
            print("Generate color point cloud.")
        elif point_type == 1:
            self.point_type = PointType.SEMANTICS_MAX
            print("Generate semantic point cloud [max fusion].")
        elif point_type == 2:
            self.point_type = PointType.SEMANTICS_BAYESIAN
            print("Generate semantic point cloud [bayesian fusion].")
        else:
            print("Invalid point type.")
            return
        # Get image size
        self.img_width, self.img_height = rospy.get_param(
            "/camera/width"
        ), rospy.get_param("/camera/height")
        # Set up CNN is use semantics
        if self.point_type is not PointType.COLOR:
            print("Setting up CNN model...")
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Get dataset
            dataset = rospy.get_param("/semantic_pcl/dataset")
            # Setup model
            model_name = "pspnet"
            model_path = rospy.get_param("/semantic_pcl/model_path")
            if dataset == "sunrgbd":  # If use version fine tuned on sunrgbd dataset
                self.n_classes = 38  # Semantic class number
                self.model = get_model(
                    model_name, self.n_classes, version="sunrgbd_res50"
                )
                state = torch.load(model_path)
                self.model.load_state_dict(state)
                self.cnn_input_size = (321, 321)
                self.mean = np.array(
                    [104.00699, 116.66877, 122.67892]
                )  # Mean value of dataset
            elif dataset == "ade20k":
                self.n_classes = 150  # Semantic class number
                self.model = get_model(model_name, self.n_classes, version="ade20k")
                state = torch.load(model_path)
                self.model.load_state_dict(
                    convert_state_dict(state["model_state"])
                )  # Remove 'module' from dictionary keys
                self.cnn_input_size = (473, 473)
                self.mean = np.array(
                    [104.00699, 116.66877, 122.67892]
                )  # Mean value of dataset
            self.model = self.model.to(self.device)
            self.model.eval()
            self.cmap = color_map(
                N=self.n_classes, normalized=False
            )  # Color map for semantic classes
        # Declare array containers
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.semantic_colors = np.zeros(
                (3, self.img_height, self.img_width, 3), dtype=np.uint8
            )  # Numpy array to store 3 decoded semantic images with highest confidences
            self.confidences = np.zeros(
                (3, self.img_height, self.img_width), dtype=np.float32
            )  # Numpy array to store top 3 class confidences
        # Set up ROS

        print("Setting up ROS...")
        self.bridge = (
            CvBridge()
        )  # CvBridge to transform ROS Image message to OpenCV image
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
        intrinsic = np.matrix(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )
        self.pcl_pub = rospy.Publisher(
            "/semantic_pcl/semantic_pcl", PointCloud2, queue_size=2
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
        self.cloud_generator = ColorPclGenerator(
            intrinsic, self.img_width, self.img_height, frame_id, self.point_type
        )
        print("Ready.")

    def color_depth_callback(self, color_img_ros, depth_img_ros):
        """
        Callback function to produce point cloud registered with semantic class color based on input color image and depth image
        \param color_img_ros (sensor_msgs.Image) the input color image (bgr8)
        \param depth_img_ros (sensor_msgs.Image) the input depth image (registered to the color image frame) (float32) values are in meters
        """
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_img_ros, desired_encoding="32FC1"
            )
            # depth_img[np.isnan(depth_img)] = 0  # Replace NaN with 0
            # depth_img[np.isinf(depth_img)] = 0  # Replace Inf with 0

        except CvBridgeError as e:
            print(e)
        x, y = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
        x, y, d = x.flatten(), y.flatten(), depth_img.flatten()
        valid_mask = ~np.isnan(d)
        x, y, d = x[valid_mask], y[valid_mask], d[valid_mask]
        x = (x - self.cx) * d / self.fx
        y = (y - self.cy) * d / self.fy
        z = d
        points = np.stack((x, y, z), axis=-1)

        # Resize depth
        if (
            depth_img.shape[0] is not self.img_height
            or depth_img.shape[1] is not self.img_width
        ):
            depth_img = resize(
                depth_img,
                (self.img_height, self.img_width),
                order=0,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True,
            )  # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)

        if self.point_type is PointType.COLOR:
            cloud_ros = self.cloud_generator.generate_cloud_color(
                color_img, depth_img, color_img_ros.header.stamp
            )
        else:
            # Do semantic segmantation
            if self.point_type is PointType.SEMANTICS_MAX:
                semantic_color, pred_confidence = self.predict_max(color_img)
                # cloud_ros = self.cloud_generator.generate_cloud_semantic_max(color_img, depth_img, semantic_color, pred_confidence, color_img_ros.header.stamp)

            elif self.point_type is PointType.SEMANTICS_BAYESIAN:
                self.predict_bayesian(color_img)
                # Produce point cloud with rgb colors, semantic colors and confidences
                # cloud_ros = self.cloud_generator.generate_cloud_semantic_bayesian(color_img, depth_img, self.semantic_colors, self.confidences, color_img_ros.header.stamp)

            # Publish semantic image
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

        # Publish point cloud
        cloud_ros = convert_to_pointcloud2(points, stamp=color_img_ros.header.stamp)
        print(cloud_ros)
        self.pcl_pub.publish(cloud_ros)

    def predict_max(self, img):
        """
        Do semantic prediction for max fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        # Take best prediction and confidence
        pred_confidence, pred_label = class_probs.max(1)
        pred_confidence = pred_confidence.squeeze(0).cpu().numpy()
        pred_label = pred_label.squeeze(0).cpu().numpy()
        pred_label = resize(
            pred_label,
            (self.img_height, self.img_width),
            order=0,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )  # order = 0, nearest neighbour
        pred_label = pred_label.astype(int)
        # Add semantic color
        semantic_color = decode_segmap(pred_label, self.n_classes, self.cmap)
        pred_confidence = resize(
            pred_confidence,
            (self.img_height, self.img_width),
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        return (semantic_color, pred_confidence)

    def predict_bayesian(self, img):
        """
        Do semantic prediction for bayesian fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img)
        # Take 3 best predictions and their confidences (probabilities)
        pred_confidences, pred_labels = torch.topk(
            input=class_probs, k=3, dim=1, largest=True, sorted=True
        )
        pred_labels = pred_labels.squeeze(0).cpu().numpy()
        pred_confidences = pred_confidences.squeeze(0).cpu().numpy()
        # Resize predicted labels and confidences to original image size
        for i in range(pred_labels.shape[0]):
            pred_labels_resized = resize(
                pred_labels[i],
                (self.img_height, self.img_width),
                order=0,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True,
            )  # order = 0, nearest neighbour
            pred_labels_resized = pred_labels_resized.astype(np.int)
            # Add semantic class colors
            self.semantic_colors[i] = decode_segmap(
                pred_labels_resized, self.n_classes, self.cmap
            )
        for i in range(pred_confidences.shape[0]):
            self.confidences[i] = resize(
                pred_confidences[i],
                (self.img_height, self.img_width),
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )

    def predict(self, img):
        """
        Do semantic segmantation
        \param img: (numpy array bgr8) The input cv image
        """
        img = (
            img.copy()
        )  # Make a copy of image because the method will modify the image
        # orig_size = (img.shape[0], img.shape[1]) # Original image size
        # Prepare image: first resize to CNN input size then extract the mean value of SUNRGBD dataset. No normalization
        img = resize(
            img,
            self.cnn_input_size,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )  # Give float64
        img = img.astype(np.float32)
        img -= self.mean
        # Convert HWC -> CHW
        img = img.transpose(2, 0, 1)
        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)  # Add batch dimension required by CNN
        with torch.no_grad():
            img = img.to(self.device)
            # Do inference
            since = time.time()
            outputs = self.model(img)  # N,C,W,H
            # Apply softmax to obtain normalized probabilities
            outputs = torch.nn.functional.softmax(outputs, 1)
            return outputs


def main(args):
    import os

    cwd = os.getcwd()

    rospy.init_node("semantic_cloud", anonymous=True)
    seg_cnn = SemanticCloud(gen_pcl=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
