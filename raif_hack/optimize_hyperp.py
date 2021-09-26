from functools import partial
import numpy as np

import hyperopt
from hyperopt import hp, fmin, tpe, atpe, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope, as_apply
from hyperopt.pyll.stochastic import sample
from hyperopt.fmin import generate_trials_to_calculate
import hpsklearn

from .model import BenchmarkModel
from .settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET, N_CV_RUNS, SEED, N_FOLDS
from .settings import SEED, N_JOBS_PER_WORKER, N_CV_RUNS, N_FOLDS, HYPEROPT_N_ROUNDS, HYPEROPT_ALGO
from .metrics import get_oof, metrics_stat
from .utils import PriceTypeEnum


def objective(space, x_train, y_train, kf):
    #     print("fit lgbm {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    # y_offer = x_train[0][x_train[0].price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
    y_manual = x_train[0][x_train[0].price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

    model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                           ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=space)
    clf_oof_train, _ = get_oof(x_train, y_train, model, kf,
                        n_cv_runs=N_CV_RUNS)

    loss_val = 0
    try:
        for clf_oof in clf_oof_train:
            # predictions_offer_cv = np.squeeze(clf_oof[x_train[0].price_type == PriceTypeEnum.OFFER_PRICE])
            # metrics = metrics_stat(y_offer.values, predictions_offer_cv / (
            #         1 + model.corr_coef))

            predictions_manual_cv = np.squeeze(clf_oof[x_train[0].price_type == PriceTypeEnum.MANUAL_PRICE])
            metrics = metrics_stat(y_manual.values, predictions_manual_cv)

            loss_val += metrics['raif_metric']
        loss_val /= len(clf_oof_train)

    except:
        # if lgmb3 fails with chosen parameters, we return maximum loss
        loss_val = 1e10

    return {'loss': loss_val, 'status': STATUS_OK}

def optimize_lgbm(X_train,
                  y_train,
                  objective_fn,
                  folds_list):
    space = {
        "importance_type": "gain",

        'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
        'seed': SEED,
        'num_threads': N_JOBS_PER_WORKER,
        'verbosity': -1,
        'max_depth': scope.int(hp.quniform("max_depth", 2, 12, 1)),
        'num_leaves': scope.int(hp.quniform('num_leaves', 20, 200, 1)),
        'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 2, 40, 1)),
        'feature_fraction': hp.uniform('feature_fraction', 0.05, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -9.2, -3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'lambda_l1': hp.loguniform('lambda_l1', -14, 7),
        'lambda_l2': hp.loguniform('lambda_l2', -14, 7),
        'n_estimators': scope.int(hp.loguniform('n_estimators', 5, 9)),
        'bagging_freq': scope.int(hp.quniform("bagging_freq", 1, 20, 1)),


        # 'device': 'gpu',
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0
    }

    mp = dict(
        n_estimators=2000,
        learning_rate=0.01,
        reg_alpha=1,
        num_leaves=40,
        min_child_samples=5,
        importance_type="gain",
        n_jobs=30,  # 1,
        random_state=563,
    )

    trials = Trials()

    X_train_stack = []
    for i in range(N_CV_RUNS):
        X_train_stack.append(X_train.copy())
    best = hyperopt.fmin(fn=partial(objective_fn,
                                    x_train=X_train_stack,
                                    y_train=y_train,
                                    kf=folds_list),
                         space=space,
                         algo=HYPEROPT_ALGO,
                         max_evals=HYPEROPT_N_ROUNDS,
                         trials=trials,
                         verbose=10,  # False
                         rstate=np.random.RandomState(SEED))


       # print("Best:", best)
       #  best_loss = trials.best_trial['result']['loss']

    best_params = space_eval(space, best)
    # best_params['nrounds'] = LGBM_NROUNDS
    print("Best params:")
    print(best_params)

    return best_params, trials