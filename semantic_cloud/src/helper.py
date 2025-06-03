import numpy as np
from numba import njit, prange
import rospy
import cv2
from sensor_msgs.msg import PointCloud2, PointField
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

