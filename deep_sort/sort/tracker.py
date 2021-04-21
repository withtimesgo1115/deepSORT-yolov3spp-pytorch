# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    # This is a multi-target tracker，preserve many tracks
    # 1. call Kalman filter to predict the new state of each track
    # 2. do matching job
    # 3. init the first frame
    # Note that when Tracker call update or predict, each track also call its own update or predict
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        # max iou，using for iou macthing
        self.max_iou_distance = max_iou_distance
        # specify cascade_depth parameter of cascade matching directly
        self.max_age = max_age
        # n_init means that it requires update at least n_init times to set the state confirmed
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()  # init a Kalman filter
        self.tracks = []  # preserve a range of tracks
        self._next_id = 1  # the next track's id

    def predict(self):
        # iterate each track and predict
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # carry out measurement update and trajectory management
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run cascade matching.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        # 1. For the result on the match
        for track_idx, detection_idx in matches:
            # track updates corresponding detection
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # 2. For the unmatched tracks, call mark_missed to mark them
        # track unmatched，if it is pending, it will be deleted, 
        # and if the update time is long, it will be deleted too
        # max age is a lifetime, the default is 70 frames
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 3. for unmatched detection， detection unmatched，do initialization
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # obtain the newest tracks，tracks marked as confirmed and tentative(pending) are saved
        # remove deleted tracks 
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # obtain all the track ids with confirmed state
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features  # Splice the tracks list into the features list
            # obtain the corresponding track id of each feature
            targets += [track.track_id for _ in track.features]
            track.features = []

        # Feature set update in distance measurement
        self.metric.partial_fit(np.asarray(features), np.asarray(targets),
                                active_targets)

    def _match(self, detections):
        # function: matching，to find the matches and unmaches
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # function：calculate the distance between track and detection，cost function
            # apply it before call KF
            # call：
            # cost_matrix = distance_metric(tracks, detections,
            #                  track_indices, detection_indices)
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            # 1. calculate the cost matrix through the nearest neighbor based on cosine distance
            cost_matrix = self.metric.distance(features, targets)

            # 2. calculate Mahalanobis distance, to get the new state matrix
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # do cascade matching，to get the matched track、unmatched track、unmatched detection
        '''
        !!!!!!!!!!!
        cascade matching
        !!!!!!!!!!!
        '''
        # gated_metric->cosine distance
        # Cascade matching is only performed on the tracks of the confirmed state
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks)

        # Combine all tracks with unconfirmed status 
        # and tracks without matching as iou_track_candidates to perform IoU matching
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1  # There was no match just now
        ]
        # unmacthed
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1  # It hasn't been matched for a long time
        ]

        '''
        !!!!!!!!!!!
        IOU matching
        IoU matching is performed on the targets that have not been successfully matched 
        in cascade matching
        !!!!!!!!!!!
        '''
        # Although the min_cost_matching is the kernel that is same as cascade matching
        # but the metric used here is iou cost which is different from the method above
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections)

        matches = matches_a + matches_b  # combine the results of two kind of matches to get the final version

        # same for unmatched
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # init at the first frame
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, detection.obj_class, self.n_init, self.max_age,
                  detection.feature))
        self._next_id += 1
