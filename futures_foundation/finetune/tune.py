"""Optuna hyperparameter search — Classifier-agnostic (concept from pipeline.tune_head).

Searches a classifier's hyperparameters scored on the VALIDATION guard (the
best_val_auc the classifier reports — test is NEVER touched), and returns params
only if they BEAT the defaults on that guard (else falls back to defaults). Used by
loop.train_loop to escape overfitting: when the walk-forward says a config does not
generalize, search for one that does.

Operates on already-featurized arrays (Xtr,ytr,Xval,yval) so featurization happens
once and any backbone plugs in via the same search.

COST NOTE: each trial is a full fit (for Mantis, a fine-tune). Keep n_trials modest
and use a max_train cap in base_kwargs — unlike the Chronos XGBoost head (seconds),
a Mantis trial is minutes.
"""
import numpy as np

from .classifier import get_classifier


def _suggest_mantis(trial):
    return dict(
        lr=trial.suggest_float('lr', 5e-5, 1e-3, log=True),
        unfreeze_blocks=trial.suggest_int('unfreeze_blocks', 1, 3),   # fewer = more reg
        new_channels=trial.suggest_int('new_channels', 6, 14),
        weight_decay=trial.suggest_float('weight_decay', 0.01, 0.3, log=True),
        batch=trial.suggest_categorical('batch', [64, 128]),
    )


def _suggest_logistic(trial):
    return dict(C=trial.suggest_float('C', 0.01, 10.0, log=True))


_DEFAULT_SUGGEST = {'mantis': _suggest_mantis, 'logistic': _suggest_logistic}


def tune(labeler, classifier, Xtr, ytr, Xval, yval, *, n_trials=10, base_kwargs=None,
         suggest=None, seed=0, verbose=True):
    """Returns dict(params, val_auc, guard_lift, generalizes). `labeler` is accepted
    for signature symmetry (featurization already done) and ignored here."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = dict(base_kwargs or {})
    suggest = suggest or _DEFAULT_SUGGEST.get(classifier)
    if suggest is None:
        raise ValueError(f"no default search space for '{classifier}'; pass suggest=")

    def val_auc(kw):
        clf = get_classifier(classifier, **kw)
        _, _, ba = clf.fit_predict(Xtr, ytr, Xval, yval, Xval, seed)   # eval=val (guard)
        return float(ba) if np.isfinite(ba) else 0.5

    base_auc = val_auc(base)

    def objective(trial):
        return val_auc(dict(base, **suggest(trial)))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    lift = study.best_value - base_auc
    params = dict(base, **study.best_params) if lift > 0 else base
    if verbose:
        print(f"  [optuna {classifier}] base_val_auc={base_auc:.4f} "
              f"best={study.best_value:.4f} lift={lift:+.4f} "
              f"{'-> use tuned' if lift > 0 else '-> keep defaults'}", flush=True)
    return dict(params=params, val_auc=study.best_value, guard_lift=lift,
                generalizes=lift > 0)
