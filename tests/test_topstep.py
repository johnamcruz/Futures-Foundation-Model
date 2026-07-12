"""Tests for the Topstep 100K combine simulator — pure function seam (issue #6).

Each test pins one acceptance criterion of the issue. Expected values are
hand-computed literals from the published contract specs, never recomputed
via the module under test.
"""
import pytest

from futures_foundation.topstep import Fill, simulate_combine

START = 100_000.0


# ---------------------------------------------------------------------------
# AC3 — friction arithmetic per symbol: a round trip of N contracts costs
# exactly N x (2 ticks + 2 x $1.40) against the gross move.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,qty,entry,exit_,expected_net",
    [
        # NQ: tick 0.25 = $5 -> $20/pt. 3 long, +10 pts: gross 600,
        # friction 3*(2*5 + 2.80) = 38.40 -> net 561.60
        ("NQ", 3, 15_000.00, 15_010.00, 561.60),
        # ES: tick 0.25 = $12.50 -> $50/pt. 2 long, +2 pts: gross 200,
        # friction 2*(2*12.50 + 2.80) = 55.60 -> net 144.40
        ("ES", 2, 4_500.00, 4_502.00, 144.40),
        # RTY: tick 0.10 = $5 -> $50/pt. 1 short, -10 pts: gross 500,
        # friction 1*(2*5 + 2.80) = 12.80 -> net 487.20
        ("RTY", -1, 2_000.00, 1_990.00, 487.20),
        # YM: tick 1.0 = $5 -> $5/pt. 4 long, +20 pts: gross 400,
        # friction 4*(2*5 + 2.80) = 51.20 -> net 348.80
        ("YM", 4, 35_000.0, 35_020.0, 348.80),
        # GC: tick 0.10 = $10 -> $100/pt. 2 long, +1.0 pt: gross 200,
        # friction 2*(2*10 + 2.80) = 45.60 -> net 154.40
        ("GC", 2, 2_400.00, 2_401.00, 154.40),
        # SI: tick 0.005 = $25 -> $5000/pt. 1 long, +0.10 pt: gross 500,
        # friction 1*(2*25 + 2.80) = 52.80 -> net 447.20
        ("SI", 1, 30.000, 30.100, 447.20),
    ],
)
def test_friction_round_trip_per_symbol(symbol, qty, entry, exit_, expected_net):
    res = simulate_combine([Fill(day=1, symbol=symbol, qty=qty, entry=entry, exit=exit_)])
    assert res.equity[-1] - START == pytest.approx(expected_net)


def test_losing_trade_charges_friction_on_top_of_gross_loss():
    # NQ 2 short, market rises 5 pts: gross -200, friction 2*12.80 = 25.60
    res = simulate_combine([Fill(day=1, symbol="NQ", qty=-2, entry=15_000.0, exit=15_005.0)])
    assert res.equity[-1] - START == pytest.approx(-225.60)


# ---------------------------------------------------------------------------
# Helpers — a YM fill engineered to land an exact net dollar P&L.
# YM: $5/pt, friction $12.80 per contract round trip (2 ticks @ $5 + $2.80).
# ---------------------------------------------------------------------------

def ym(day, net):
    """One-contract YM fill whose net P&L is exactly `net` dollars."""
    points = (net + 12.80) / 5.0
    return Fill(day=day, symbol="YM", qty=1, entry=40_000.0, exit=40_000.0 + points)


# ---------------------------------------------------------------------------
# AC1 — terminal states: target hit -> passed; data exhausted -> timeout.
# ---------------------------------------------------------------------------

def test_target_hit_passes():
    # Two balanced days totaling exactly the $6,000 target (best day 50%).
    res = simulate_combine([ym(1, 3_000.0), ym(2, 3_000.0)])
    assert res.state == "passed"
    assert res.days == 2
    assert res.equity[-1] == pytest.approx(START + 6_000.0)


def test_pass_stops_processing_further_fills():
    # The fill after the pass must not be applied; equity path ends at the pass.
    res = simulate_combine([ym(1, 3_000.0), ym(2, 3_000.0), ym(3, -1_000.0)])
    assert res.state == "passed"
    assert res.days == 2
    assert len(res.equity) == 2
    assert res.equity[-1] == pytest.approx(START + 6_000.0)


def test_data_exhausted_is_timeout():
    res = simulate_combine([ym(1, 500.0), ym(2, 500.0), ym(3, -200.0)])
    assert res.state == "timeout"
    assert res.days == 3
    assert res.equity[-1] == pytest.approx(START + 800.0)
