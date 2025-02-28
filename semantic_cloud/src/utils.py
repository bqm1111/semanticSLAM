#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf.transformations as tf_trans

import numpy as np
import cv2
from numba import njit, prange


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

    # Pass 1: Compute bounding box
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
                min_u = min(min_u, u)
                max_u = max(max_u, u)
                min_v = min(min_v, v)
                max_v = max(max_v, v)
    
    # Pass 2: Map depth to color frame and apply bounding box
    for y in prange(h):  # Again, only outer loop parallelized
        for x in range(w):  # Inner loop sequential
            z = depth_meters[y, x]
            if z <= 0:
                continue

            xd = (x - cx_d) * inv_fx_d * z
            yd = (y - cy_d) * inv_fy_d * z

            X = r11 * xd + r12 * yd + r13 * z + t1
            Y = r21 * xd + r22 * yd + r23 * z + t2
            Z = r31 * xd + r32 * yd + r33 * z + t3

            if Z <= 0:
                continue

            u = int(X * fx_c / Z + cx_c)
            v = int(Y * fy_c / Z + cy_c)

            if min_u <= u <= max_u and min_v <= v <= max_v:  # Inside bounding box
                depth_color_aligned[v, u] = Z

    return depth_color_aligned


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

        print(
            "================> Depth to Color Transformation Matrix:\n",
            depth_to_color_matrix,
        )
        return depth_to_color_matrix

    except tf2_ros.LookupException:
        rospy.logerr(f"Transform between {source_frame} and {target_frame} not found.")
    except tf2_ros.ExtrapolationException:
        rospy.logerr("TF extrapolation error.")
    except rospy.ROSInterruptException:
        pass
