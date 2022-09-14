# vim: expandtab:ts=4:sw=4
import numpy as np
from typing import Optional, Dict
import cattrs
from sa.ais.ais_reader import get_ship_type
from sa.sa_utils.dataclass import DeepsortTrackInfo, TargetOutputData
from .detection import Detection
from .kalman_filter import KalmanFilter


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center x and bottom y of the bounding box,
    `a` is the aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    track_info: Optional[DeepsortTrackInfo]
        track information this track originates from.


    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    track_info: DeepsortTrackInfo
        voyage information related to track

    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        n_init_dao_relation: int,
        max_age: Dict[str, int],
        feature: Optional[np.ndarray] = None,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.track_info = DeepsortTrackInfo(id=track_id)

        self._n_init = n_init
        self._max_age = max_age
        self._match_dao_target_history = np.ones(n_init_dao_relation, np.int32) * -1

    def to_tlwh(self) -> np.ndarray:
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[0] -= ret[2] / 2
        ret[1] -= ret[3]
        return ret

    def to_tlbr(self) -> np.ndarray:
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf: KalmanFilter) -> None:
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        if self.is_confirmed() and not self.global_matched:
            self._match_dao_target_history = np.roll(self._match_dao_target_history, -1)
            self._match_dao_target_history[-1] = -1

    def update(self, kf: KalmanFilter, detection: Detection) -> None:
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def refresh_track_info(self, dao_target: TargetOutputData) -> None:
        track_info_dict = {
            "id": self.track_id,
            "global_id": dao_target.id,
            "source": list(dao_target.sid.keys()),
            "lat": dao_target.lat,
            "lon": dao_target.lon,
            "width": dao_target.width,
            "length": dao_target.length,
            "object_type": get_ship_type(int(dao_target.ship_type)),
            "mmsi": int(dao_target.mmsi),
            "vn": dao_target.vn,
            "ve": dao_target.ve,
            "sog": dao_target.sog,
            "cog": dao_target.cog,
        }
        self.track_info = cattrs.structure(track_info_dict, DeepsortTrackInfo)

    def mark_match_dao_target(self, dao_target: TargetOutputData) -> None:
        self._match_dao_target_history[-1] = dao_target.id
        if (self._match_dao_target_history == dao_target.id).all():
            self.refresh_track_info(dao_target)

    def mark_missed(self) -> None:
        """Mark this track as missed (no association at the current time step)."""
        if (
            self.state == TrackState.Tentative
            and self.time_since_update > self._max_age["tentative"]
        ):
            self.state = TrackState.Deleted
        elif (
            self.global_matched
            and self.time_since_update > self._max_age["global_matched"]
        ):
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age["confirmed"]:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    @property
    def global_matched(self) -> bool:
        return self.track_info.global_id > 0

    @property
    def global_id(self) -> int:
        return self.track_info.global_id
