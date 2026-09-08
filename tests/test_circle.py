"""Tests for the circular statistics in ``pltstat.circle``.

The plotting functions of the module are covered by ``test_plots.py``; this
module checks the conversions and the directional mean and deviation.
"""

import numpy as np
import pytest

from pltstat import circle


class TestConversions:
    """Conversion between a value on a cyclic scale and radians."""

    @pytest.mark.parametrize("value", [0, 6, 12, 18])
    def test_round_trip(self, value):
        assert circle.rad2val(circle.val2rad(value)) == pytest.approx(value)

    def test_full_turn_maps_to_high(self):
        """A full turn comes back as `high`, not as 0.

        Both denote the same point on the circle; the conversion keeps the
        upper end of the scale rather than wrapping it to the lower one.
        """
        assert circle.rad2val(circle.val2rad(24)) == pytest.approx(24)

    def test_val2rad_half_turn(self):
        assert circle.val2rad(12) == pytest.approx(np.pi)

    def test_rad2val_half_turn(self):
        assert circle.rad2val(np.pi) == pytest.approx(12)

    def test_custom_high(self):
        assert circle.val2rad(180, high=360) == pytest.approx(np.pi)


class TestCircularMean:
    """Directional mean on a cyclic scale."""

    def test_mean_of_identical_values(self):
        assert circle.mean(np.array([6.0, 6.0, 6.0])) == pytest.approx(6.0)

    def test_mean_wraps_around_midnight(self):
        """The mean of 23:00 and 01:00 is midnight, not noon."""
        result = circle.mean(np.array([23.0, 1.0]))
        assert result == pytest.approx(0.0, abs=1e-6) or result == pytest.approx(
            24.0, abs=1e-6
        )

    def test_antipodal_values_have_undefined_mean(self):
        """Opposite points cancel out, so the mean is not defined."""
        assert np.isnan(circle.mean(np.array([0.0, 12.0])))


class TestCircularStd:
    """Directional standard deviation on a cyclic scale."""

    def test_identical_values_have_zero_std(self):
        assert circle.std(np.array([6.0, 6.0, 6.0])) == pytest.approx(0.0, abs=1e-9)

    def test_spread_values_have_positive_std(self):
        assert circle.std(np.array([1.0, 7.0, 13.0, 19.0])) > 0
