import os
from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import LabelEncoder

__all__ = ['load_adult', 'load_cover_type', 'load_diabetes',
           'load_mammography', 'load_oil', 'load_phoneme', 'load_satimage']


def load_adult(only_numeric=True, dropna=True):
    """Load the adult dataset.

    Parameters
    ----------
    only_numeric : bool
        Whether to return a dataframe containing only the numeric
        features. By default, only numerical features are returned.
    dropna : bool
        Whether to drop rows containing NA values.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    NUMERIC_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capitalgain',
                       'capitalloss', 'hoursperweek']
    df = pd.read_csv(os.path.join('data', 'adult.csv'), na_values=['?'])

    if dropna:
        df = df.dropna()

    target = df['class']
    data = df.drop(columns='class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    if only_numeric:
        return data[NUMERIC_COLUMNS], target
    return data, target


def load_cover_type():
    """Load the Forest cover type dataset.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    data, target = fetch_covtype(return_X_y=True)

    # select only the class 3 and 4
    mask = (target == 3) | (target == 4)
    data = data[mask]
    target = target[mask]

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target


def load_diabetes():
    """Load the Pima Indian Diabetes.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    df = pd.read_csv(os.path.join('data', 'diabetes.csv'))

    target = df['class']
    data = df.drop(columns='class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target


def load_mammography():
    """Load the mammography dataset.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    df = pd.read_csv(os.path.join('data', 'mammography.csv'))

    target = df['class']
    data = df.drop(columns='class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target


def load_oil():
    """Load the oil dataset.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    df = pd.read_csv(os.path.join('data', 'oil.csv'))

    target = df['class']
    data = df.drop(columns='class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target


def load_phoneme():
    """Load the phoneme dataset.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    df = pd.read_csv(os.path.join('data', 'phoneme.csv'))

    target = df['Class']
    data = df.drop(columns='Class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    return data, target


def load_satimage():
    """Load the satellite image datasets.

    Returns
    -------
    data : dataframe, shape (n_samples, n_features)
        A dataframe containing the data.
    target : ndarray, shape (n_samples,)
        The target label associated to the data.
    """
    df = pd.read_csv(os.path.join('data', 'satimage.csv'))

    target = df['class']
    data = df.drop(columns='class')

    encoder = LabelEncoder()
    target = encoder.fit_transform(target)

    # find the minority class
    target_counter = Counter(target)
    minority_class = min(target_counter, key=target_counter.get)
    mask_minority_class = target == minority_class
    target[mask_minority_class] = 0
    target[~mask_minority_class] = 1

    return data, target
