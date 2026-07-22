"""Stage-2.6 pretext: NEXT-LEG forecasting — the seq2seq family, prediction unit = THE TREND LEG.

From a context window ending at a CONFIRMED fractal pivot, predict the market's own next pattern
element in BARS (user spec 2026-07-16 — bar-counts are instrument- and TF-agnostic, the fractal
unit; NO ATR anywhere — ATR is derived/volatile, raw candles only): (1) how many bars the newborn
leg runs, (2) how many bars the counter-leg (the retest) lasts. Pivots = the PURE fractal detector
(k-bar swing high/low comparisons on raw candles, detect_fractal_pivots — no ATR filter), legs =
alternating swings (keep-first). Targets come from that same deterministic structure applied to
future price — pure self-supervision, zero labels, zero strategy, zero derived indicators.
Direction is NOT predicted (legs alternate: known at confirm — degenerate target).

Why this survives where turn-electra/lc512 died: it stays a FORECASTER (generic objective, the
input/prediction aimed at the pattern unit — the banked design law), keeps the candle-seq2seq
ANCHOR (mse_weight, the anti-drift piece proven load-bearing), warm-starts from the promoted
base with early blocks frozen (the reorder recipe). GATE: the trend-lifecycle scorecard — the
new checkpoint must BEAT the banked per-direction probes (0.7689/0.7635 start, 0.7519/0.7603
end) BEFORE any WR pipeline touches it; the probe gate here is the anti-shortcut forecast gate.
"""
from .forecast import ForecastTask


class NextLegTask(ForecastTask):
    name, trainer = 'nextleg', 'train_ssl_nextleg'

    @property
    def control_contract(self):
        return 'nextleg_forecast_and_leg_skill_v1'

    @staticmethod
    def _metric(history_row, report_name, trainer_name=None):
        value = history_row.get(report_name)
        if value is None and trainer_name is not None:
            value = history_row.get(trainer_name)
        return None if value is None else float(value)

    def control_evidence(self, history_row, probe_res):
        """Measure what NextLeg added, not generic knowledge inherited from Seq2Seq.

        Shuffle/random adapters retain almost all parent representation, so their generic linear
        probes can tie REAL even when their next-leg targets are unlearnable. REAL must instead
        beat every corruption on persistence skill and both future leg-duration correlations.
        The ordinary representation gate remains independently mandatory to catch forgetting.
        """
        return {
            'forecast_skill': self._metric(history_row, 'forecast_skill', 'skill'),
            'leg_corr1': self._metric(history_row, 'leg_corr1'),
            'leg_corr2': self._metric(history_row, 'leg_corr2'),
        }

    def compare_control_evidence(self, real, controls):
        metrics = ('forecast_skill', 'leg_corr1', 'leg_corr2')
        margins = {
            name: {
                metric: (None if real.get(metric) is None or row.get(metric) is None else
                         float(real[metric]) - float(row[metric]))
                for metric in metrics
            }
            for name, row in controls.items()
        }
        real_positive = all(real.get(metric) is not None and float(real[metric]) > 0
                            for metric in metrics)
        passed = bool(real_positive and controls and all(
            margin is not None and margin > 0
            for row in margins.values() for margin in row.values()))
        shuffle = margins.get('shuffle') or {}
        return passed, margins, shuffle.get('forecast_skill')

    def reserve(self, cfg):
        # context + enough future for BOTH legs to RESOLVE (unresolved -> masked, not fabricated).
        # LEAK FIX (2026-07-17): the target reads the pivot TWO ahead (t2 = o_nn - o_n), so the
        # future span the target touches is up to t1 + t2 <= 2*leg_cap, NOT one leg_cap. Reserving
        # only 1*leg_cap let boundary anchors' t2 read across the train/val and pre-holdout/2026
        # split (window_starts only enforces contiguity over `reserve` bars). Reserve BOTH legs so
        # every kept anchor's o_nn stays inside its own split. Asserted in the trainer.
        return max(int(x) for x in cfg['context_lengths']) + 2 * int(cfg.get('leg_cap', 256))

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        verdict = super().finalize_verdict(verdict, fc_skill, probe_res)
        verdict['pretext_note'] = ('next-leg forecaster (bars): judge on the trend-lifecycle '
                                   'scorecard FIRST (must beat the banked 0.75-0.77 per-direction '
                                   'probes), then the standard WR pipeline; leg-skill is a '
                                   'learning diagnostic only')
        return verdict
