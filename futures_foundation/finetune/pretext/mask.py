"""Stage-1 pretext: BERT-style masked modeling (UNCHANGED). Gate = REAL encodes regime/vol/
structure better than vanilla (mean_core_delta > margin) and doesn't collapse."""
from .base import PretextTask


class MaskTask(PretextTask):
    name, trainer = 'mask', 'train_ssl_mask'

    def _decide(self, probe_res, no_collapse, margin, dir_margin, detail):
        stream_win_rate = probe_res.get('stream_win_rate')
        broad_market_context = stream_win_rate is None or float(stream_win_rate) >= 0.5
        detail['broad_market_context'] = broad_market_context
        return bool(probe_res['mean_core_delta'] > margin and broad_market_context
                    and no_collapse), detail
