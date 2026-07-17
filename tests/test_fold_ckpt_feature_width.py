"""A changed FEATURE WIDTH must invalidate the fold checkpoint.

Regression guard: classifier + kwargs + seed + fold layout were identical between an
embedding-only run (0 handcraft cols) and a handcraft run (17), so the signature collided
and the stale folds reloaded — the WF verdict came back byte-identical and the new feature
set was never fitted.
"""
from futures_foundation.finetune.resume import config_signature

CLF = 'mantis_frozen'
CK = {'head': 'logistic', 'batch': 256, 'raw_C': 5, 'raw_seq': 64}
FOLDS = [[3674, 1196, 1225], [3609, 1225, 1336]]


def _sig(width):
    return config_signature(CLF, CK, 0, FOLDS, [width])   # per-row feature shape


def test_feature_width_changes_the_signature():
    assert _sig(0) != _sig(17), 'embedding-only and handcraft runs must not share fold state'


def test_signature_is_stable_for_an_unchanged_config():
    assert _sig(17) == _sig(17)


if __name__ == '__main__':
    test_feature_width_changes_the_signature()
    test_signature_is_stable_for_an_unchanged_config()
    print('ok')
