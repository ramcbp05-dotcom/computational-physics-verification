"""Tests for the error-report checker.

The checker decides whether a diff in the committed error report is a real
regression or machine noise, so getting it wrong is costly in both directions:
too strict and CI fails on every runner until people stop reading it, too loose
and a weakened bound slips through. These tests pin both edges.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_error_report",
    Path(__file__).resolve().parents[1] / "scripts" / "check_error_report.py",
)
check = importlib.util.module_from_spec(_SPEC)
sys.modules["check_error_report"] = check
_SPEC.loader.exec_module(check)


HEADER = "# Numerical error report\n"


def report(*rows: str) -> str:
    return HEADER + "".join(row + "\n" for row in rows)


class TestAgree:
    def test_identical_values_agree(self):
        ok, how = check.agree(1.0, 1.0)
        assert ok and how == "identical"

    def test_two_round_off_measurements_agree_however_far_apart(self):
        """The case that made a byte-exact diff useless.

        Both numbers are reports of the same exact zero on different hardware.
        They differ by 23%, and that is not information about the physics.
        """
        ok, how = check.agree(2.205e-14, 1.795e-14)
        assert ok, f"expected round-off values to agree, got: {how}"
        assert "round-off" in how

    def test_round_off_rule_applies_even_across_an_order_of_magnitude(self):
        ok, _ = check.agree(2.220e-16, 1.628e-13)
        assert ok, "both sit below the round-off ceiling, so both are the same zero"

    def test_a_discretisation_error_agrees_within_tolerance(self):
        ok, how = check.agree(2.932e-10, 2.933e-10)
        assert ok, how

    def test_a_weakened_bound_fails(self):
        """Leapfrog energy drift moving two orders of magnitude is a real change."""
        ok, how = check.agree(2.682e-05, 1.0e-03)
        assert not ok
        assert "exceeds" in how

    def test_the_tolerance_edge_is_where_it_says_it_is(self):
        assert check.agree(1.0, 1.0 - 0.019)[0], "1.9% should pass a 2% tolerance"
        assert not check.agree(1.0, 1.0 - 0.021)[0], "2.1% should fail it"

    def test_one_value_above_the_round_off_ceiling_is_not_noise(self):
        """A quantity that was round-off and is now 1e-6 has genuinely moved."""
        ok, _ = check.agree(1e-15, 1e-6)
        assert not ok


class TestExactZeroSpelling:
    """The bug the first version of the checker shipped with.

    The generator prints an exact zero as the words ``0 (exact)`` and anything
    else as a number. Whether a round-off quantity lands on exactly 0.0 is
    machine-dependent, so one machine writes ``0 (exact)`` where another writes
    ``1.332e-16``. Comparing text before normalising, that reads as a changed
    heading rather than as round-off, and CI reported a stale report on three
    Python versions when nothing was stale.
    """

    def test_words_and_a_round_off_number_are_the_same_measurement(self):
        old = report("| 0.3 | 4 | 0.833654607012 | 1.332e-16 |")
        new = report("| 0.3 | 4 | 0.833654607012 | 0 (exact) |")
        failures, tolerated = check.compare(old, new)
        assert failures == [], "\n".join(failures)
        assert len(tolerated) == 1 and "round-off" in tolerated[0]

    def test_it_works_in_the_other_direction_too(self):
        old = report("| x | 0 (exact) |")
        new = report("| x | 2.136e-16 |")
        failures, _ = check.compare(old, new)
        assert failures == []

    def test_words_on_both_sides_agree(self):
        text = report("| x | 0 (exact) |")
        failures, tolerated = check.compare(text, text)
        assert failures == [] and tolerated == []

    def test_an_exact_zero_becoming_a_real_error_still_fails(self):
        """The rule must not swallow a quantity that genuinely stopped being zero."""
        old = report("| x | 0 (exact) |")
        new = report("| x | 1.000e-05 |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1 and "exceeds" in failures[0]

    def test_a_trailing_newline_is_not_a_structural_change(self):
        """`error_report.py --print` adds one, the committed file does not."""
        failures, _ = check.compare(report("| a | 1.0 |"), report("| a | 1.0 |") + "\n")
        assert failures == []


class TestCompare:
    def test_an_unchanged_report_passes(self):
        text = report("| a | 1.000e-05 |")
        failures, tolerated = check.compare(text, text)
        assert failures == [] and tolerated == []

    def test_noise_is_reported_as_tolerated_not_as_a_failure(self):
        old = report("| dL/L | 2.205e-14 |")
        new = report("| dL/L | 1.795e-14 |")
        failures, tolerated = check.compare(old, new)
        assert failures == []
        assert len(tolerated) == 1 and "round-off" in tolerated[0]

    def test_a_real_bound_movement_fails(self):
        old = report("| leapfrog dE/E | 2.682e-05 |")
        new = report("| leapfrog dE/E | 1.000e-03 |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1
        assert "exceeds" in failures[0]

    def test_changed_text_is_caught_as_staleness(self):
        """The bug this checker was written after.

        The generator's column heading was fixed but the committed report was
        never regenerated, so the repository shipped a table whose header read
        ``Boris rel |v| drift``, and in Markdown those pipes are column
        separators, so the table rendered broken.
        """
        old = report("| steps | Boris rel |v| drift |")
        new = report("| steps | Boris rel. speed drift |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1
        assert "stale" in failures[0]

    def test_text_staleness_is_caught_even_when_numbers_are_identical(self):
        old = report("### Boris push: |v| conservation vs step size", "| x | 1.0e-15 |")
        new = report("### Boris push: speed conservation vs step size", "| x | 1.0e-15 |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1 and "stale" in failures[0]

    def test_a_structural_change_fails_before_comparing_lines(self):
        old = report("| a | 1.0 |")
        new = report("| a | 1.0 |", "| b | 2.0 |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1
        assert "changed length" in failures[0]

    def test_the_provenance_line_is_exempt(self):
        """It carries a timestamp and the NumPy version; both always differ."""
        old = report(
            "Generated by `scripts/error_report.py` on 2026-11-01 05:59 UTC "
            "with NumPy 2.5.4."
        )
        new = report(
            "Generated by `scripts/error_report.py` on 2026-11-02 11:04 UTC "
            "with NumPy 2.6.0."
        )
        failures, tolerated = check.compare(old, new)
        assert failures == [] and tolerated == []

    def test_a_number_appearing_where_text_was_is_a_failure(self):
        old = report("| a | exact identity |")
        new = report("| a | 1.0e-05 |")
        failures, _ = check.compare(old, new)
        assert len(failures) == 1


class TestAgainstTheRealReport:
    def test_the_committed_report_is_current(self):
        """Regenerating must reproduce the committed report within tolerance.

        This is the same check CI runs. It is here too so that a stale report
        fails an ordinary test run, not only the workflow. It regenerates
        through ``--print``, so running the suite leaves the working tree
        clean: an earlier version wrote over the committed file and made
        every test run show a one-line timestamp diff.
        """
        path = Path(__file__).resolve().parents[1] / "results" / "numerical-error-report.md"
        assert path.is_file(), f"{path} is missing"

        before = path.read_bytes()
        failures, _ = check.compare(path.read_text(), check.regenerate())
        assert path.read_bytes() == before, "checking the report must not rewrite it"
        assert failures == [], "report is stale or a bound moved:\n" + "\n".join(failures)
