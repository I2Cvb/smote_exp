import os
from collections import defaultdict
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import safe_indexing

from datasets import load_adult
from datasets import load_cover_type
from datasets import load_diabetes
from datasets import load_mammography
from datasets import load_oil
from datasets import load_phoneme
from datasets import load_satimage


def _fit_score(pipe, param_grid, X, y, train_idx, test_idx, cv_idx):
    """Fit a pipeline and score.

    Parameters
    ----------
    pipe : Estimator
        A scikit-learn pipeline.
    param_grid : ParameterGrid
        A ParameterGrid with all the parameters to try for the pipeline.
    X : ndarray, shape (n_samples, n_features)
        The full dataset.
    y : ndarray, shape (n_samples,)
        The associated target.
    train_idx : ndarray, (n_train_samples,)
        The training indexes.
    test_idx : ndarray, (n_test_samples,)
        The testing indexes.
    cv_idx : int
        The index of the fold.
    Returns
    -------
    cv_results : dict
        A dictionary containing the score and parameters.
    """
    cv_results = defaultdict(list)
    X_train, y_train = safe_indexing(X, train_idx), y[train_idx]
    X_test, y_test = safe_indexing(X, test_idx), y[test_idx]

    for param in param_grid:
        multiplier = param['smote__sampling_strategy']
        def resampling(y):
            counter = Counter(y)
            minority_class = min(counter, key=counter.get)
            return {minority_class: int(counter[minority_class] +
                                        counter[minority_class] * multiplier)}
        param['smote__sampling_strategy'] = resampling
        pipe_cv = clone(pipe)
        pipe_cv.set_params(**param)
        pipe_cv.fit(X_train, y_train)
        y_pred = pipe_cv.predict_proba(X_train)
        train_score = roc_auc_score(y_train, y_pred[:, 1])
        y_pred = pipe_cv.predict_proba(X_test)
        test_score = roc_auc_score(y_test, y_pred[:, 1])

        cv_results['cv_idx'].append(cv_idx)
        cv_results['train_score'].append(train_score)
        cv_results['test_score'].append(test_score)
        for k, v in param.items():
            if k == 'smote__sampling_strategy':
                cv_results[k].append(multiplier)
            else:
                cv_results[k].append(v)

    return cv_results


def _merge_dicts(d1, d2):
    """Merge two dictionaries."""
    for k in d1.keys():
        d1[k] += d2[k]
    return d1

for name, func_dataset, factor in [
        ('adult', load_adult, [0.5, 1, 2, 3, 4, 5]),
        ('cover_type', load_cover_type, [0.5, 1, 2, 3, 4, 5]),
        ('diabetes', load_diabetes, [0.5, 1, 2, 3, 4, 5]),
        ('mammography', load_mammography, [0.5, 1, 2, 3, 4, 5]),
        ('oil', load_oil, [0.5, 1, 2, 3, 4, 5]),
        ('phoneme', load_phoneme, [0.5, 1, 2, 3, 4, 5]),
        ('satimage', load_satimage, [0.5, 1, 2, 3, 4 , 5])]:
    X, y = func_dataset()
    pipe = make_pipeline(SMOTE(random_state=42),
                         DecisionTreeClassifier(random_state=42))
    param_grid = ParameterGrid(
        {'smote__sampling_strategy': np.array(factor)}
    )
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    results = Parallel(n_jobs=-1)(
        delayed(_fit_score)(pipe, param_grid, X, y,
                            train_idx, test_idx, cv_idx)
        for cv_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)))

    cv_results = results[-1]
    for res in results[:-1]:
        cv_results = _merge_dicts(cv_results, res)

    if not os.path.exists('results'):
        os.makedirs('results')

    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(os.path.join('results', name + '.csv'))
