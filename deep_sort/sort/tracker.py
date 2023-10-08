# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from typing import Tuple, List, Set, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular
from sa.sa_utils.dataclass import TargetOutputData
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection


class Tracker:
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
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

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

    def __init__(
        self,
        metric: NearestNeighborDistanceMetric,
        max_iou_distance: float = 0.7,
        max_age: Dict[str, int] = {
            "tentative": 0,
            "confirmed": 70,
            "global_matched": 70,
        },
        n_init: int = 3,
        n_init_dao_relation: int = 5,
        kf_only_position: bool = False,
        kf_std_position: float = 1.0 / 20,
        kf_std_velocity: float = 1.0 / 160,
        kf_std_position_dao_target: float = 1.0 / 20,
    ) -> None:
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.n_init_dao_relation = n_init_dao_relation

        self.kf = kalman_filter.KalmanFilter(
            only_position=kf_only_position,
            std_weight_position=kf_std_position,
            std_weight_velocity=kf_std_velocity,
        )
        self.kf_only_position = kf_only_position
        self.kf_dao_measurement_cov = np.diag(
            np.square([kf_std_position_dao_target, kf_std_position_dao_target])
        )
        self.tracks: List[Track] = []
        self._next_id = 1

    def predict(self) -> None:
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: List[Detection]) -> None:
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        def gated_metric(
            tracks: List[Track],
            dets: List[Detection],
            track_indices: List[int],
            detection_indices: List[int],
        ) -> np.ndarray:
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                only_position=self.kf_only_position,
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            max(self.max_age.values()),
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection) -> None:
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.n_init_dao_relation,
                self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1

    @property
    def occupied_global_ids(self) -> Set[int]:
        return {track.global_id for track in self.tracks if track.global_matched}

    def update_dao_relations(
        self,
        unoccupied_dao_targets_xy: np.ndarray,
        unoccupied_dao_targets: List[TargetOutputData],
        occupied_dao_targets_dict: Dict[int, TargetOutputData],
        gating_threshold: float = kalman_filter.chi2inv95[2],
        gated_cost: float = 1e5,
    ) -> None:
        if not unoccupied_dao_targets and not occupied_dao_targets_dict:
            return
        unmatched_confirmed_tracks: List[int] = []
        unmatched_confirmed_tracks_tlbr: List[np.ndarray] = []
        unmatched_confirmed_tracks_cov: List[np.ndarray] = []
        for idx, track in enumerate(self.tracks):
            # refresh former matched tracks
            if track.global_matched:
                if track.global_id in occupied_dao_targets_dict:
                    track.refresh_track_info(occupied_dao_targets_dict[track.global_id])
            elif track.is_confirmed():
                unmatched_confirmed_tracks.append(idx)
                unmatched_confirmed_tracks_tlbr.append(track.to_tlbr())
                unmatched_confirmed_tracks_cov.append(track.covariance[0:2, 0:2])
        if not unoccupied_dao_targets or not unmatched_confirmed_tracks:
            return
        # cost matrix generation
        cost_matrix_list: List[np.ndarray] = []
        for idx, tlbr in enumerate(unmatched_confirmed_tracks_tlbr):
            cov = unmatched_confirmed_tracks_cov[idx]
            gating_distance = self._calculate_track2dao_maha_distance(
                tlbr, cov, unoccupied_dao_targets_xy
            )
            gating_distance[gating_distance > gating_threshold] = gated_cost + 1e-5
            cost_matrix_list.append(gating_distance)
        # linear assignment
        cost_matrix = np.array(cost_matrix_list)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < gated_cost:
                track = self.tracks[unmatched_confirmed_tracks[row]]
                dao_target = unoccupied_dao_targets[col]
                track.mark_match_dao_target(dao_target)

    def _calculate_track2dao_maha_distance(
        self,
        tlbr: np.ndarray,
        cov: np.ndarray,
        measurements: np.ndarray,
        track_cov_considered: bool = False,
    ) -> np.ndarray:
        if track_cov_considered:
            cholesky_factor = np.linalg.cholesky(cov + self.kf_dao_measurement_cov)
        else:
            cholesky_factor = np.linalg.cholesky(self.kf_dao_measurement_cov)
        x1, y1, x2, y2 = tlbr
        x = measurements[:, 0]
        y = measurements[:, 1]
        d_x = np.minimum(np.abs(x - x1), np.abs(x - x2))
        d_x[(x <= x2) & (x >= x1)] = 0
        d_y = np.minimum(np.abs(y - y1), np.abs(y - y2))
        d_y[(y <= x2) & (y >= y1)] = 0
        d = np.array([d_x, d_y]).T
        z = solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        maha_distance = np.sum(z * z, axis=0)
        return maha_distance