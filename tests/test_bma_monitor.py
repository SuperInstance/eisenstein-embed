"""Tests for BMA drift detection monitor."""

import pytest

from eisenstein_embed.bma_monitor import BMAMonitor


class TestBMAMonitor:
    def test_no_drift_with_few_samples(self):
        mon = BMAMonitor()
        for _ in range(5):
            mon.record("q", 0.8, "semantic")
        assert not mon.is_drift_detected()

    def test_baseline_established(self):
        mon = BMAMonitor(min_samples=10)
        for i in range(10):
            mon.record(f"q{i}", 0.8, "semantic")
        assert mon._baseline is not None
        assert mon._baseline == pytest.approx(0.8, abs=0.01)

    def test_drift_detected_when_scores_drop(self):
        mon = BMAMonitor(min_samples=10, drift_threshold=0.1)
        # Establish baseline at 0.8
        for i in range(10):
            mon.record(f"q{i}", 0.8, "semantic")
        # Drop scores
        for i in range(10):
            mon.record(f"q_drop{i}", 0.5, "semantic")
        assert mon.is_drift_detected()

    def test_no_drift_when_scores_stable(self):
        mon = BMAMonitor(min_samples=10, drift_threshold=0.1)
        for i in range(20):
            mon.record(f"q{i}", 0.8, "semantic")
        assert not mon.is_drift_detected()

    def test_suggest_threshold(self):
        mon = BMAMonitor(min_samples=10)
        for i in range(10):
            mon.record(f"q{i}", 0.8, "semantic")
        thresh = mon.suggest_threshold()
        assert 0.1 <= thresh <= 0.9

    def test_reset_baseline(self):
        mon = BMAMonitor(min_samples=10, drift_threshold=0.1)
        for i in range(10):
            mon.record(f"q{i}", 0.8, "semantic")
        for i in range(10):
            mon.record(f"q_drop{i}", 0.5, "semantic")
        assert mon.is_drift_detected()
        mon.reset_baseline()
        assert not mon.is_drift_detected()
        assert mon._baseline == pytest.approx(0.5, abs=0.05)

    def test_stats(self):
        mon = BMAMonitor(min_samples=5)
        for i in range(5):
            mon.record(f"q{i}", 0.7, "semantic")
        stats = mon.stats
        assert stats["count"] == 5
        assert stats["baseline"] == pytest.approx(0.7, abs=0.01)
        assert stats["alerts"] == 0
