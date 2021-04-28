#!/usr/bin/env python
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT
"""

from collections import deque
import numpy as np
from skimage import io, draw
import copy
from scipy.spatial import distance
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, Imu
import message_filters
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
from darknet_custom import *
from cv_bridge import CvBridge, CvBridgeError
from distance_assistant.msg import BboxMsg, DetectionMsg, DetectionsMsg
import temporal_filter
import datetime


def parse_array_from_string(list_str, dtype=int):
    """ Create a 1D array from text in string.
    Args:
        list_str: input string holding the array elements.
            Array elements should be contained in brackets [] and seperated
            by comma.
        dtype: data type of the array elements. Default is "int"
    Returns:
        1D numpy array
    """
    list_str = list_str.lstrip().rstrip()
    if not (list_str.startswith('[') and list_str.endswith(']')):
        msg = 'list_str should start with "[" and end with "]".'
        raise (SyntaxError(msg))

    return np.array(list_str[1:-1].split(','), dtype=dtype)


class DistanceAssistant:
    def __init__(self):
        """Initializes the ros node."""
        self.last_log_scene = {}
        self.prev_time = time.time()
        self.init_params()
        self.person_detector = PersonDetector()
        if self.enable_temporal_filter:
            self.temporal_filter = temporal_filter.PersonTemporalFilter(
                self.max_age, self.min_hits)
        self.bridge = CvBridge()
        self.init_subscribers()
        self.init_publishers()

    def init_subscribers(self):
        """Initialize ROS topic subscriptions."""
        # Camera Info
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info',
                                                CameraInfo,
                                                self.camera_info_callback)
        # Synchronized RGB + Depth
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                                  Image)
        self.depth_sub = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.TimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.rgb_depth_callback)

        if self.auto_calibrate:
            self.imu_sub = rospy.Subscriber('/camera/accel/sample',
                                            Imu,
                                            self.imu_callback,
                                            queue_size=1)
            self.depth_sub.registerCallback(self.compute_camera_height)

    def init_publishers(self):
        """Initialize node's publication topics."""
        self.vis_img_pub = rospy.Publisher("/distance_assistant/vis_img",
                                           Image,
                                           queue_size=5)
        self.detections_pub = rospy.Publisher("/distance_assistant/detections",
                                              DetectionsMsg,
                                              queue_size=5)

    def init_params(self):
        """Initialize node parameters from ROS parameter server."""
        self.confidence_thr = rospy.get_param('~confidence_thr')
        self.distance_thr = rospy.get_param('~distance_thr')
        self.max_distance = rospy.get_param('~max_distance')
        self.publish_vis = rospy.get_param('~publish_vis')
        self.show_distances = rospy.get_param('~show_distances')
        self.show_cross_distances = rospy.get_param('~show_cross_distances')
        self.draw_bbox = rospy.get_param('~draw_bbox')
        self.draw_circle = rospy.get_param('~draw_circle')

        # Calibration parameters
        self.auto_calibrate = rospy.get_param('~auto_calibrate')
        self.num_depth_samples = rospy.get_param('~num_depth_samples')
        self.num_imu_samples = rospy.get_param('~num_imu_samples')
        self.filter_weight = rospy.get_param('~filter_weight')
        self.min_camera_height = rospy.get_param('~min_camera_height')
        self.max_camera_height = rospy.get_param('~max_camera_height')
        self.calibration_roi = parse_array_from_string(
            rospy.get_param('~calibration_roi'))
        self.imu_frame_counter = 0
        self.depth_frame_counter = 0
        self.init_camera_intrinsic = False
        self.init_camera_height = False
        self.init_camera_rotation = False
        self.accel = np.array([0, 0, 0])
        self.camera_height = 0

        if not self.auto_calibrate:
            self.read_manual_calibration_params()

        # Temporal filter parameters
        self.enable_temporal_filter = rospy.get_param(
            '~enable_temporal_filter')
        self.min_hits = rospy.get_param('~min_hits')
        self.max_age = rospy.get_param('~max_age')

    def read_manual_calibration_params(self):
        """ Read precomputed camera extrics parameters from file """
        self.camera_roll = rospy.get_param('~camera_roll') * np.pi / 180.
        self.camera_pitch = rospy.get_param('~camera_pitch') * np.pi / 180.
        self.camera_height = rospy.get_param('~camera_height')
        self.R = self.compute_rotation_matrix(self.camera_pitch,
                                              self.camera_roll)
        self.init_camera_rotation = True
        self.init_camera_height = True

    def camera_info_callback(self, camera_info):
        """Initialize the camera intrinsic parameteres from camera info topic."""
        self.width = camera_info.width
        self.height = camera_info.height
        self.camera_info_K = np.array(camera_info.K).reshape([3, 3])
        self.camera_info_D = np.array(camera_info.D)
        self.rgb_frame_id = camera_info.header.frame_id
        self.init_camera_intrinsic = True

    def compute_camera_height(self, depth_msg):
        if self.init_camera_height:  # height is already initialized
            return
        if not self.init_camera_intrinsic:
            rospy.logwarn_throttle(
                3, 'Waiting for camera intrinsics to be initialized')
            return
        if not self.init_camera_rotation:
            rospy.logwarn_throttle(
                3, 'Waiting for camera rotation matrix to be initialized')
            return

        depth_img = self.bridge.imgmsg_to_cv2(depth_msg,
                                              desired_encoding="passthrough")
        depth_img = depth_img.astype('float32') / 1000.

        # Compute point cloud
        pcd = cv2.rgbd.depthTo3d(depth_img, self.camera_info_K)

        # Filter points within the calibration ROI
        pcd = pcd[self.calibration_roi[1]:self.calibration_roi[3],
                  self.calibration_roi[0]:self.calibration_roi[2]].reshape(
                      -1, 3)
        roi_size = pcd.shape[0]

        # Filter invalid depth points
        pcd = pcd[np.where(pcd[:, 2] > 0)]

        # Check if there are enough depth data
        if float(len(pcd)) / roi_size < 0.2:
            rospy.logwarn_throttle(
                2, 'Skipping frame - there were not enough ground depth'
                ' points to compute height. Make sure that camera height is'
                ' within allowed limits and there are no large obstacles'
                ' in front of the camera.')
            return

        # Compute the height values
        heights = np.dot(pcd, self.R[:, 1]).squeeze()

        # Compute height distribution between min & max camera heights with
        # 10 cm resolution
        hist, bins = np.histogram(
            heights,
            np.arange(self.min_camera_height, self.max_camera_height, 0.1))

        # filter out outlier points and compute height
        mode_index = np.argmax(hist)
        heights = heights[np.where((heights > bins[mode_index])
                                   & (heights < bins[mode_index + 1]))]
        height = np.median(heights)

        # compute moving average
        self.camera_height = (height * self.filter_weight +
                              (1 - self.filter_weight) * self.camera_height)
        self.depth_frame_counter += 1

        # Check if we had enough samples to finalize depth measurement
        if self.depth_frame_counter > self.num_depth_samples:
            msg = 'Initialized camera height to: {0} meters'
            rospy.loginfo(msg.format(self.camera_height))
            self.init_camera_height = True

    def imu_callback(self, imu_msg):
        """Computes camera pitch & roll wrt to gravity vector.

        Assumes camera is stationary (i.e. platform is not moving) and
        computes the camera roll & pitch angles from linear accelerations
        (note that when the camera is stationary any accelerations are
        due to gravitational forces).
        Realsense IMU orientation angles and acceleration vectors share the
        coordinate system with the depth sensor where
          - The positive x-axis points to the right.
          - The positive y-axis points down.
          - The positive z-axis points forward

        References:
            https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/

        """
        if self.init_camera_rotation:  # camera rotation is already initialized
            return

        current_accel = np.array([
            imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # compute moving average
        self.accel = (self.filter_weight * current_accel +
                      (1 - self.filter_weight) * self.accel)

        self.imu_frame_counter += 1
        if self.imu_frame_counter > self.num_imu_samples:
            self.camera_pitch = -np.arctan2(
                self.accel[2], np.linalg.norm([self.accel[0], self.accel[1]]))
            self.camera_roll = np.arctan2(
                self.accel[0], np.linalg.norm([self.accel[2], self.accel[1]]))

            msg = ('Initialized camera rotation matrix with \n'
                   'pitch: {0} degrees\n'
                   'roll: {1} degrees\n')

            rospy.loginfo(
                msg.format(self.camera_pitch * 180 / np.pi,
                           self.camera_roll * 180 / np.pi))

            self.R = self.compute_rotation_matrix(self.camera_pitch,
                                                  self.camera_roll)
            self.init_camera_rotation = True
            self.imu_sub.unregister(
            )  # TODO: continously monitor IMU for changes

    def compute_rotation_matrix(self, pitch=0, roll=0, yaw=0):
        """ Computes and returns the 3x3 camera rotation matrix

        Arguments:
            pitch: camera rotation around x axis (radians)
            roll: camera rotation around z axis (radians)
            yaw: camera rotation around y axis (radians)

        Returns:
            3x3 Camera rotation matrix
        """

        # Rotation around camera's x axis (pointing right)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])

        # Rotation around camera's y axis (pointing dawn)
        Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0],
                       [np.sin(yaw), 0, np.cos(yaw)]])

        # Rotation around camera's z axis (pointing forward)
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))
        return R

    def sample_circle_points(self, center_x, center_y, radius, num_samples=8):
        """ sample points regularly around perimeter of a circle

        Arguments:
            center_x : x coordinate of circle's center
            center_y : y coordinate of circle's center
            radius: radius of the circle
            num_samples: number of samples that will be returned

        Returns:
            [Nx2] array of points samples around the circle perimeter
        """

        points = []
        for theta in np.arange(0, np.pi * 2, np.pi * 2 / num_samples):
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            points.append([x, y])

        return np.array(points)

    def run_person_detector(self, cv_img):
        """ Wrapper method around the person detector.

        Arguments:
            cv_img: [HxWx3] RGB image

        Returns:
            List of detections (TODO: create a detection class)
        """
        return self.person_detector.detect(cv_img, self.confidence_thr)

    def compute_person_centroids(self, depth_img, detections):
        """ Computes person positions w.r.t. to the camera frame and updates
        the detections with computed [x,y,z] position and distance to camera

        Arguments:
            depth_img: 2D depth image (uint16)
                where each value is distance from camera in mm
            detections: list of person detections ([Detection])

        Returns:
            None
        """
        pcd = cv2.rgbd.depthTo3d(depth_img, self.camera_info_K)
        if pcd is None:
            return None

        for detection in detections:
            bbox = np.array(detection.bbox)
            person_pcd = pcd[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            x = np.median(person_pcd[:, :, 0])
            y = np.median(person_pcd[:, :, 1])
            z = np.median(person_pcd[:, :, 2])
            detection.centroid = [x, y, z]
            detection.distance_to_cam = distance.euclidean(
                detection.centroid, [0, 0, 0])
        return

    def draw_distances(self, vis_img, detections):
        """ Write distances on the image.

        Arguments:
            vis_img: RGB image that will be modified
            detections: list of person detections

        Returns:
            vis_img: image with distance text
        """
        for detection in detections:
            bbox = np.array(detection.bbox)

            text_position = tuple([self.width - bbox[2], bbox[1]])
            color = detection.color.value
            if self.show_cross_distances:
                if detection.proximity != np.inf:
                    cv2.putText(vis_img, '{:.2f}'.format(detection.proximity),
                                text_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color, 2, cv2.LINE_AA)
            else:
                cv2.putText(vis_img,
                            '{:.2f}'.format(detection.distance_to_cam),
                            text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            2, cv2.LINE_AA)

        return vis_img

    def draw_bboxes(self, vis_img, detections):
        """ Draw bounding boxes around people in the image.

        Arguments:
            vis_img: RGB image that will be modified
            detections: list of person detections

        Returns:
            vis_image: image with color-coded bounding boxes
        """
        for detection in detections:
            bbox = list(detection.bbox)
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          detection.color.value, 3)
        return vis_img

    def draw_circles(self, vis_img, detections):
        """Draw projected circles in the visualization image.

        Arguments:
            vis_img: RGB image that will be modified
            detections: list of person detections

        Returns:
            vis_img: image with color-coded circles around people
        """
        mask = np.zeros((vis_img.shape[0], vis_img.shape[1], 3), np.uint8)
        for _, detection in enumerate(detections):
            position_cam = np.array(detection.centroid)

            # rotate centroid to axis align with ground plane
            position_world = np.dot(position_cam, self.R)

            circle_points_world = self.sample_circle_points(
                position_world[0], position_world[2], 1,
                32)  # circle points on x,z plane
            circle_points_world = np.insert(circle_points_world,
                                            1,
                                            self.camera_height,
                                            axis=1)  # set y axis

            # rotate back to camera frame
            circle_points_cam = np.dot(circle_points_world,
                                       np.linalg.inv(self.R))

            # compute pixel values
            pixels = np.zeros((circle_points_cam.shape[0], 2))
            pixels[:, 0] = (circle_points_cam[:, 0] / circle_points_cam[:, 2] *
                            self.camera_info_K[0, 0] +
                            self.camera_info_K[0, 2]).astype(int)
            pixels[:, 1] = (circle_points_cam[:, 1] / circle_points_cam[:, 2] *
                            self.camera_info_K[1, 1] +
                            self.camera_info_K[1, 2]).astype(int)

            # draw ellipse from the points
            ellipse = cv2.fitEllipse(pixels.astype('float32'))
            cv2.ellipse(mask, ellipse, detection.color.value, 10, cv2.LINE_AA)

        # clear bound box regions for occlusion visualization
        for _, detection in enumerate(detections):
            bbox = np.array(detection.bbox)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        # update visualization image
        vis_img[np.where(mask[:, :, 0])] = Color.RED.value
        vis_img[np.where(mask[:, :, 1])] = Color.GREEN.value
        return vis_img

    def publish_detections(self, detections, compute_time, timestamp):
        """ Publishes list of detections.

        Arguments:
            detections: list of people detections
            timestamp: RGB image capture timestamp

        Returns:
            None
        """
        msg = DetectionsMsg()
        msg.header.stamp = timestamp
        msg.header.frame_id = self.rgb_frame_id
        for detection in detections:
            detection_msg = DetectionMsg()
            detection_msg.name = "person"
            detection_msg.confidence = detection.score
            bbox = BboxMsg()
            bbox.top_left_x = detection.bbox.top_left_x
            bbox.top_left_y = detection.bbox.top_left_y
            bbox.bottom_right_x = detection.bbox.bottom_right_x
            bbox.bottom_right_y = detection.bbox.bottom_right_y
            detection_msg.bbox = bbox
            detection_msg.centroid = detection.centroid
            detection_msg.instance_id = detection.instance_id
            msg.detections.append(detection_msg)

        msg.camera_intrinsics = self.camera_info_K.reshape(
            1, 9).squeeze().tolist()
        msg.camera_rotation_matrix = self.R.reshape(1, 9).squeeze().tolist()
        msg.camera_height = self.camera_height
        msg.compute_time = compute_time
        self.detections_pub.publish(msg)

    def perform_actions(self, nr_detections_dict):
        if nr_detections_dict["RED"] > 0:
            # touch the sound sentinel file
            with open("/tmp/sounds/alerts", "w") as f:
                f.write("")

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        log_line = "%s RED: %d GREEN: %d\n" % (ts, nr_detections_dict["RED"], nr_detections_dict["GREEN"])

        # avoid writing the same thing non stop
        # only log when the number of detected
        # people changes, or changes state
        if nr_detections_dict != self.last_log_scene:
            with open("/tmp/log/da.log", "a+") as f:
                self.last_log_scene = nr_detections_dict
                f.write(log_line)

    def compute_proximities(self, detections):
        """ For each person in detections computes the closest proximity to
        the other people in the scene and updates the detection attributes for
        proximity and color.

        Arguments:
            detections: list of people detections ([Detection])
        Returns:
            None
        """

        N = len(detections)
        distances = np.ones((N, N)) * np.inf
        nr_detections_dict = {
            "RED": 0,
            "GREEN": 0,
        }

        for i in range(N):
            for j in range(i + 1, N):
                distances[i, j] = distance.euclidean(detections[i].centroid,
                                                     detections[j].centroid)
                distances[j, i] = distances[i, j]

            detections[i].proximity = np.min(distances[i, :])

            if detections[i].proximity < self.distance_thr:
                detections[i].color = Color.RED
                nr_detections_dict["RED"] += 1
            else:
                detections[i].color = Color.GREEN
                nr_detections_dict["GREEN"] += 1

        self.perform_actions(nr_detections_dict)

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        """Callback method for syncronized RGB + Depth image topics.

        Runs the person detector on the RGB image,
        checks proximity for each detection and publishes a
        visualization image indicating social distancing for detected people.

        Arguments:
            rgb_msg: ROS RGB message
            depth_msg: ROS Depth message time-synchronized RGB message

        Returns:
            None
        """
        start_time = time.time()

        if not (self.init_camera_intrinsic and self.init_camera_rotation
                and self.init_camera_height):
            rospy.logwarn_throttle(
                3, 'Camera parameters are not initialized yet')
            return

        rospy.logdebug("Rate: %.2f ", 1 / (time.time() - self.prev_time))
        self.prev_time = time.time()
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        try:
            #Convert the depth image using the default passthrough encoding
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
            depth_img = depth_img.astype('float32') / 1000.
        except CvBridgeError as e:
            print(e)

        orig_img = copy.deepcopy(cv_img)
        detections = self.run_person_detector(cv_img)
        self.compute_person_centroids(depth_img, detections)

        # filter out detections beyond sensor range
        detections = [
            detection for detection in detections
            if detection.distance_to_cam < self.max_distance
        ]

        if self.enable_temporal_filter:
            detections = self.temporal_filter.update(detections, self.height,
                                                     self.width)

        # compute proximities for each person in the scene
        self.compute_proximities(detections)
        if (self.publish_vis):
            vis_img = np.copy(orig_img)
            if self.draw_circle:
                vis_img = self.draw_circles(vis_img, detections)
            if self.draw_bbox:
                vis_img = self.draw_bboxes(vis_img, detections)
            vis_img = np.copy(np.fliplr(vis_img))  # mirror the image
            if self.show_distances:
                vis_img = self.draw_distances(vis_img, detections)

            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="rgb8")
            # keep the original timestamp
            vis_msg.header.stamp = rgb_msg.header.stamp
            self.vis_img_pub.publish(vis_msg)

        compute_time = time.time() - start_time
        self.publish_detections(detections, compute_time, rgb_msg.header.stamp)


def main(args):
    rospy.init_node('distance_assistant', anonymous=True)
    distance_assistant = DistanceAssistant()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
