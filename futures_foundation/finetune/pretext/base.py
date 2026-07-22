"""PretextTask — base class for a pluggable SSL pretraining objective (a "stage").

Subclass to add a new pretrain experiment WITHOUT editing the orchestrator (ssl.py). Each task
owns four things: how much window to RESERVE per window, how to TRAIN (which _ssl_torch trainer),
its report-only GATE on the probe, and any pretext-specific verdict fields. Trainers swallow
unknown kwargs (**_ignore), so the shared cfg is safe to pass to any task.
"""


class PretextTask:
    name = 'base'
    trainer = None                                        # _ssl_torch trainer fn name
    requires_related_series = False
    requires_stream_layout = False

    def reserve(self, cfg):
        """Extra bars reserved per window beyond `seq` (0 = none)."""
        return 0

    def train(self, big, tr, va, cfg, control):
        """-> (best_encoder_state, history) under a control ('real'|'shuffle'|'random')."""
        from .. import _ssl_torch                          # lazy: keep module load torch-free
        kw = {k: v for k, v in cfg.items() if k != 'pretext'}
        return getattr(_ssl_torch, self.trainer)(big, tr, va, control=control, **kw)

    def gate(self, probe_res, std, margin, dir_margin, forecast_skill=None):
        """Report-only gate on the PROBE (representation content), NOT the loss. Builds the shared
        detail dict, then defers the pass/fail to `_decide`. ``forecast_skill`` is available to
        forecasting subclasses; representation-only stages deliberately ignore it."""
        no_collapse = bool(std > 0.01)
        detail = {'no_collapse': no_collapse}
        if probe_res is None:
            return no_collapse, {**detail, 'probe': None}
        detail.update({'mean_core_delta': float(probe_res['mean_core_delta']),
                       'descriptive_delta': float(probe_res.get('descriptive_delta', 0.0)),
                       'fwd_absmove_delta': float(probe_res.get('fwd_absmove_delta', 0.0)),
                       'fwd_dir_delta': float(probe_res.get('fwd_dir_delta', 0.0)),
                       'forward_score': float(probe_res.get('forward_score', 0.0)),
                       'stream_win_rate': probe_res.get('stream_win_rate'),
                       'average_target_win_rate': probe_res.get('average_target_win_rate'),
                       'worst_stream_win_rate': probe_res.get('worst_stream_win_rate'),
                       'learns_regime_vol_structure': bool(probe_res['learns_regime_vol_structure'])})
        return self._decide(probe_res, no_collapse, margin, dir_margin, detail)

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        raise NotImplementedError

    def finalize_verdict(self, verdict, fc_skill, probe_res):
        """Add any pretext-specific fields to the saved verdict (default: none)."""
        return verdict

    @property
    def control_contract(self):
        """Evidence contract used to prove REAL learned more than corrupted controls."""
        return 'representation_probe_v1'

    def control_evidence(self, history_row, probe_res):
        """Extract control-comparable evidence for this objective.

        Representation objectives use Probe Atlas lift. Forecasting objectives may override this
        because a warm-started corrupted adapter retains its parent's generic representation even
        when it cannot learn the stage's temporal target.
        """
        return {'mean_core_delta': (None if probe_res is None else
                                    float(probe_res['mean_core_delta']))}

    def compare_control_evidence(self, real, controls):
        """Return (pass, per-control margins, temporal-order margin)."""
        value = real.get('mean_core_delta')
        margins = {
            name: (None if value is None or row.get('mean_core_delta') is None else
                   float(value) - float(row['mean_core_delta']))
            for name, row in controls.items()
        }
        passed = bool(value is not None and controls and
                      all(margin is not None and margin > 0 for margin in margins.values()))
        return passed, margins, margins.get('shuffle')
