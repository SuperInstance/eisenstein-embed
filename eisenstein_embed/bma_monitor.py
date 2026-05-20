"""BMA drift detection — know when embeddings go stale."""

from typing import List, Optional
from collections import deque

import numpy as np


class BMAMonitor:
    """Monitors retrieval outcomes and detects drift in match quality.

    Maintains a rolling window of match scores. Drift is detected when
    the recent mean drops significantly below the historical mean,
    indicating that embeddings may have gone stale (e.g., vocabulary
    shift, concept drift).
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.15,
        min_samples: int = 20,
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self._scores: deque = deque(maxlen=window_size)
        self._baseline: Optional[float] = None
        self._alert_count = 0

    def record(self, query: str, best_score: float, layer: str) -> None:
        """Record the outcome of a match operation."""
        self._scores.append(best_score)
        if self._baseline is None and len(self._scores) >= self.min_samples:
            self._baseline = float(np.mean(list(self._scores)))

    def is_drift_detected(self) -> bool:
        """Return True if recent scores suggest embedding drift."""
        if len(self._scores) < self.min_samples:
            return False
        if self._baseline is None:
            return False

        recent = list(self._scores)[-self.min_samples:]
        recent_mean = float(np.mean(recent))
        drop = self._baseline - recent_mean

        if drop > self.drift_threshold:
            self._alert_count += 1
            return True
        return False

    def suggest_threshold(self) -> float:
        """Suggest a dynamic threshold based on recent score distribution.

        Returns a threshold set at one standard deviation below the
        recent mean, clamped to sensible bounds.
        """
        if len(self._scores) < self.min_samples:
            return 0.5
        recent = list(self._scores)[-self.min_samples:]
        mean = float(np.mean(recent))
        std = float(np.std(recent))
        suggested = mean - std
        return float(np.clip(suggested, 0.1, 0.9))

    def reset_baseline(self) -> None:
        """Recalibrate the baseline to the recent window mean."""
        if len(self._scores) >= self.min_samples:
            recent = list(self._scores)[-self.min_samples:]
            self._baseline = float(np.mean(recent))
            self._alert_count = 0

    @property
    def stats(self) -> dict:
        """Return current monitoring statistics."""
        scores_list = list(self._scores)
        return {
            "count": len(scores_list),
            "baseline": self._baseline,
            "recent_mean": float(np.mean(scores_list[-self.min_samples:])) if scores_list else None,
            "alerts": self._alert_count,
        }
