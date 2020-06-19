"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT
"""
import numpy as np
import rospy
from scipy.spatial import distance
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from darknet_custom import Detection, Bbox, Centroid


class KalmanFilterManager(object):
    """ Initializes a wrapper around the Kalman Filter for multiple handling
    multiple filter instances and their associated IDs

        Arguments:
            x: 1x15 state vector
               [score, x, y, z, tlx, tly, brx, bry,
               vx, vy, vz, vtlx, vtly, vbrx, vbry]
            dt: time between updates (currently assumes fixed frame rate)
    """

    instance_id = 0
    max_instance_id = 20

    def __init__(self, x, dt=1.0):
        self.dim_x = 15  # number of state variables
        self.dim_z = 8  # number of measured variables
        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        # initialize state transition matrix

        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, dt],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # initialize measurement matrix (8x15)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

        # initialize state covariance matrix
        self.kf.P[self.dim_z:, self.dim_z:] *= 100
        self.kf.P *= 10  # set initial covariance higher for hidden state

        # Set process covariance matrix
        self.kf.Q[self.dim_z:, self.dim_z:] *= 0.01

        self.kf.x[:self.dim_z] = x.reshape(self.dim_z, 1)

        self.time_since_update = 0
        self.hits = 0
        self.continuing_hits = 0
        self.id = KalmanFilterManager.instance_id
        KalmanFilterManager.instance_id += 1
        KalmanFilterManager.instance_id = np.mod(
            KalmanFilterManager.instance_id, self.max_instance_id)

    def predict(self, dt=1):
        """ Run Kalman Filter prediction step."""
        self.kf.predict()
        if self.time_since_update > 0:  # there was missed detections
            self.continuing_hits = 0
        self.time_since_update += 1
        return self.kf.x[:self.dim_z].squeeze()

    def update(self, x):
        """ Run Kalman Filter update step."""
        self.time_since_update = 0
        self.hits += 1
        self.continuing_hits += 1
        self.kf.update(x)

    def get_state(self):
        """ Returns the current state estimate."""
        return self.kf.x[:self.dim_z].squeeze()


def compute_iou(bboxA, bboxB):
    """ Computes the intersection over union between two bounding boxes.

    Arguments:
        bboxA, bboxB: Bounding Boxes

    Returns:
        iou (float): intersection over union between bboxA and bboxB
    """
    # find coordinates of intersecting rectangle
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of rectangles
    boxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    boxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def associate_detections_to_filters(detections,
                                    filters,
                                    max_distance=0.8,
                                    iou_weight=0.5):
    """ Computes distance matrix between detections and filters and associates
    detections and filters via Hungarian algorithm.

    Arguments:
        detections_matrix (Nx8): measured state of the detected people
        filters (Mx8): predicted state of the detected people

    Returns:
        matches (Kx2) : indexes of matching predictions and filters
        unmatched_detections (1 x N-K):  indexes of unmatched detections
        unmatched_filters (1 x M-K): indexes of unmatched filters

    """
    if (len(filters) == 0):
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)),
                np.empty((0), dtype=int))

    dist_matrix = np.zeros((len(detections), len(filters)), dtype=np.float32)
    iou_matrix = np.zeros_like(dist_matrix)
    for d, det in enumerate(detections):
        for t, trk in enumerate(filters):
            dist_matrix[d, t] = distance.euclidean(det[0:3], trk[0:3])
            iou_matrix[d, t] = compute_iou(det[3:7], trk[3:7])

    cost_matrix = dist_matrix - iou_weight * iou_matrix

    matched_indices = linear_assignment(cost_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_filters = []
    for t, trk in enumerate(filters):
        if (t not in matched_indices[:, 1]):
            unmatched_filters.append(t)

    #filter out matched with distance > max_distance
    matches = []
    for m in matched_indices:
        if (dist_matrix[m[0], m[1]] > max_distance):
            unmatched_detections.append(m[0])
            unmatched_filters.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_filters)


class PersonTemporalFilter(object):
    """Uses Kalman filter to smooth detection noise across video frames.

    Arguments:
        max_age (default=5): maximum age of the filter before it's being dropped
        min_hits (default=10): minimum number of hits needed
            before a filter instance is considered valid.
    """
    def __init__(self, max_age=5, min_hits=10):
        self.max_age = max_age
        self.min_hits = min_hits
        self.filters = []
        self.frame_count = 0
        rospy.loginfo('Initialized Person Temporal Filter')

    def update(self, detections, img_height, img_width):
        # parse detections matrix (Nx8)
        detections_matrix = np.array([
            [detection.score] + list(detection.centroid) + list(detection.bbox)
            for detection in detections
        ])

        if self.frame_count < self.min_hits:
            self.frame_count += 1

        # run Kalman prediction and get predicted states
        filters = np.zeros((len(self.filters), 8))
        invalid_filters = []  # indexes of filters that will be removed

        for t, filter_instance in enumerate(filters):
            prediction = self.filters[t].predict().reshape((1, -1))
            filter_instance[:] = prediction
            if (np.any(np.isnan(prediction))):
                invalid_filters.append(t)

        # remove invalid states from the list
        for index in invalid_filters:
            self.filters.pop(index)

        # associate detections with filters
        matches, unmatched_dets, unmatched_filters = associate_detections_to_filters(
            detections_matrix, filters)

        # update matched filters with assigned detections
        for t, filter_instance in enumerate(self.filters):
            if t not in unmatched_filters:
                d = matches[np.where(matches[:, 1] == t)[0],
                            0]  # matched detection index
                filter_instance.update(detections_matrix[d, :])

        # create new filters for unmatched detections
        for i in unmatched_dets:
            filter_instance = KalmanFilterManager(detections_matrix[i, :])
            self.filters.append(filter_instance)
            rospy.logdebug('Created a new temporal filter with id:',
                           filter_instance.instance_id)

        # Get filtered state from the Kalman filter for active states and return them
        result = []  # filtered detections with updated state information
        index = len(self.filters)
        for filter_instance in reversed(self.filters):
            state_vec = filter_instance.get_state()
            # Check if the frame is valid
            if ((filter_instance.time_since_update < self.max_age)
                    and (filter_instance.hits >= self.min_hits
                         or self.frame_count < self.min_hits)):
                score = state_vec[0]
                centroid = Centroid(*state_vec[1:4])
                bbox = state_vec[4:8].astype(int)
                min_coordinates = np.array([0, 0, 0, 0])
                max_coordinates = np.array([
                    img_width - 1, img_height - 1, img_width - 1,
                    img_height - 1
                ])
                bbox = np.max((bbox, min_coordinates), axis=0)
                bbox = np.min((bbox, max_coordinates), axis=0)
                detection = Detection(score, Bbox(*bbox), centroid)
                detection.instance_id = filter_instance.instance_id
                detection.distance_to_cam = distance.euclidean(
                    detection.centroid, [0, 0, 0])
                result.append(detection)
            index -= 1
            #remove dead filter instance
            if (filter_instance.time_since_update >= self.max_age):
                self.filters.pop(index)
                rospy.logdebug('Detection lost for: ',
                               filter_instance.instance_id)
        return result
