import typing
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from .settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, CATEGORICAL_STE_FEATURES, \
    TARGET
from .utils import PriceTypeEnum

import copy
from functools import partial

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

N_CV_RUNS = 1


def get_oof(x_train, y_train, model, kf,
            n_cv_runs=N_CV_RUNS):
    est_list = []
    ntrain = x_train[0].shape[0]
    oof_train = np.zeros((n_cv_runs, ntrain,))

    for cv_run_id in range(n_cv_runs):
        est_list.append([])
        for i, (train_index, test_index) in enumerate(kf[cv_run_id]):
            x_tr = x_train[cv_run_id].iloc[train_index]
            y_tr = y_train.iloc[train_index]
            x_te = x_train[cv_run_id].iloc[test_index]

            X_offer_tr = x_tr[x_tr.price_type == PriceTypeEnum.OFFER_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
            y_offer_tr = x_tr[x_tr.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
            X_manual_tr = x_tr[x_tr.price_type == PriceTypeEnum.MANUAL_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
            y_manual_tr = x_tr[x_tr.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

            X_manual_te = x_te[x_te.price_type == PriceTypeEnum.MANUAL_PRICE][
                NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]

            clf_loc = copy.deepcopy(model)
            clf_loc.fit(X_offer_tr, y_offer_tr, X_manual_tr, y_manual_tr)

            est_list[cv_run_id].append(clf_loc)

            oof_pred_loc = clf_loc.predict(
                x_train[cv_run_id][NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES].iloc[test_index])
            oof_train[cv_run_id, test_index] = oof_pred_loc

    return oof_train.reshape(n_cv_runs, -1, 1), est_list


def deviation_metric_one_sample(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9


def deviation_metric(y_true: np.array, y_pred: np.array) -> float:
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()


def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    return np.median(np.abs(y_pred - y_true) / y_true)


def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {'mape': mape, 'mdape': mdape, 'rmse': rmse, 'r2': r2, 'raif_metric': raif_metric}


EPS = 1e-8
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0.9, 1.8, 2.7, 3.6, 4.5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.1, 2.2, 3.3, 4.4, 5.5])) <= EPS
assert deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.15, 2.3, 3.45, 4.6, 5.75])) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1.3, 2.6, 3.9, 5.2, 6.5])) - 1) <= EPS
assert np.abs(
    deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0.7, 1.4, 2.1, 2.8, 3.5])) - 1 * NEGATIVE_WEIGHT) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50])) - 9) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0])) - 9 * NEGATIVE_WEIGHT) <= EPS
assert np.abs(deviation_metric(np.array([1, 2, 3, 4, 5]), np.array([1, 2.2, 3.3, 5, 50])) - 85 / 45) <= EPS
