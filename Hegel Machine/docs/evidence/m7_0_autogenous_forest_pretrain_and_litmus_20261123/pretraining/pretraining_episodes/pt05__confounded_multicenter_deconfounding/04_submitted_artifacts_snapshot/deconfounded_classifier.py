"""Prevalence-neutral classification of an aligned multicenter panel."""

import numpy as np
from sklearn.linear_model import LogisticRegression

_SIGNAL = np.array([2, 5, 9, 14, 21])


def _align(X, centers):
    """Estimate each scanner's shared offset and gain from control features."""
    X = np.asarray(X, dtype=float)
    aligned = np.empty((len(X), len(_SIGNAL)))
    control = np.ones(X.shape[1], dtype=bool)
    control[_SIGNAL] = False
    for center in np.unique(centers):
        mask = centers == center
        panel = X[mask]
        refs = panel[:, control]
        offset = np.nanmedian(np.nanmean(refs, axis=0))
        gain = np.nanmedian(np.nanstd(refs, axis=0))
        aligned[mask] = (panel[:, _SIGNAL] - offset) / gain
    return aligned


def _neutral_weights(y, centers):
    """Give each observed class equal weight within each training center."""
    weights = np.empty(len(y), dtype=float)
    for center in np.unique(centers):
        mask = centers == center
        for label in np.unique(y[mask]):
            cell = mask & (y == label)
            weights[cell] = 1.0 / np.count_nonzero(cell)
    return weights / weights.mean()


def _basis(aligned, fill):
    a, b, d, e, f = np.where(np.isnan(aligned), fill, aligned).T
    # An affine correction of the product induces both constituent main effects.
    return np.column_stack((a - b, d * e, f, d, e))


def fit_and_predict_probabilities(X_train, y_train, center_train, X_test, center_test) -> np.ndarray:
    """Predict positive-class probabilities under equal class priors.

    Scanner alignment uses the unlabeled panel from each test center. The
    center identifier is not a predictor and test prevalence is not estimated.
    """
    y = np.asarray(y_train).reshape(-1)
    train_centers = np.asarray(center_train).reshape(-1)
    test_centers = np.asarray(center_test).reshape(-1)
    if len(test_centers) == 0:
        return np.empty(0, dtype=float)
    train = _align(X_train, train_centers)
    test = _align(X_test, test_centers)
    weights = _neutral_weights(y, train_centers)
    observed = np.isfinite(train)
    support = weights @ observed
    fill = np.divide(weights @ np.where(observed, train, 0.0), support,
                     out=np.zeros(len(_SIGNAL)), where=support > 0)
    model = LogisticRegression(C=3.0, max_iter=1000)
    model.fit(_basis(train, fill), y, sample_weight=weights)
    return model.predict_proba(_basis(test, fill))[:, 1]
