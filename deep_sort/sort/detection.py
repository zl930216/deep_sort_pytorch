# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray
        A feature vector that describes the object contained in this image.

    """

    def __init__(
        self, tlwh: np.ndarray, confidence: float, feature: np.ndarray
    ) -> None:
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self) -> np.ndarray:
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert bounding box to format `(center x, bottom y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[0] += ret[2] / 2
        ret[1] += ret[3]
        ret[2] /= ret[3]
        return ret
