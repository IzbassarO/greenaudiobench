"""M7 analysis: the Pareto dominance rule must be exact, not approximate."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_results import pareto_front  # noqa: E402


def front(rows):
    df = pd.DataFrame(rows, columns=["benefit", "cost"])
    return pareto_front(df, "benefit", "cost").tolist()


def test_strictly_dominated_point_is_excluded():
    # (0.9, 2.0) is worse on both axes than (0.95, 1.0)
    assert front([(0.95, 1.0), (0.90, 2.0)]) == [True, False]


def test_trade_off_points_both_survive():
    # cheaper but less accurate vs pricier and more accurate
    assert front([(0.92, 0.70), (0.98, 0.73)]) == [True, True]


def test_equal_benefit_higher_cost_is_dominated():
    assert front([(0.96, 1.4), (0.96, 5.0)]) == [True, False]


def test_equal_cost_lower_benefit_is_dominated():
    assert front([(0.97, 1.0), (0.93, 1.0)]) == [True, False]


def test_identical_points_are_both_kept():
    # neither strictly beats the other, so neither may be silently dropped
    assert front([(0.96, 1.0), (0.96, 1.0)]) == [True, True]


def test_matches_brute_force_on_the_official_efficiency_values():
    """Cross-check against an independent O(n^2) implementation."""
    df = pd.DataFrame({
        "benefit": [0.9615, 0.9720, 0.9225, 0.9615, 0.9795],
        "cost": [5.0134, 1.4605, 0.7061, 1.3651, 0.7255],
    })
    got = pareto_front(df, "benefit", "cost").tolist()
    brute = []
    for i in range(len(df)):
        a = df.iloc[i]
        brute.append(not any(
            (df.iloc[j]["benefit"] >= a["benefit"] and df.iloc[j]["cost"] <= a["cost"]
             and (df.iloc[j]["benefit"] > a["benefit"] or df.iloc[j]["cost"] < a["cost"]))
            for j in range(len(df)) if j != i))
    assert got == brute
    assert got == [False, False, True, False, True]  # PANNs + MS-CLAP
