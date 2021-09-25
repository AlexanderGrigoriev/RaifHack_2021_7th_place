import argparse
import logging.config

import numpy as np
import pandas as pd
from traceback import format_exc
from sklearn.model_selection import cross_val_score, KFold

from raif_hack.model import BenchmarkModel
from raif_hack.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET, N_CV_RUNS, SEED, N_FOLDS
from raif_hack.utils import PriceTypeEnum
from raif_hack.metrics import metrics_stat, get_oof
from raif_hack.features import prepare_categorical, clean_values

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для обучения модели
     
     Примеры:
        1) с poetry - poetry run python3 train.py --train_data /path/to/train/data --model_path /path/to/model
        2) без poetry - python3 train.py --train_data /path/to/train/data --model_path /path/to/model
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--train_data", "-d", type=str, dest="d", required=True, help="Путь до обучающего датасета")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True, help="Куда сохранить обученную ML модель")

    return parser.parse_args()



if __name__ == "__main__":

    try:
        logger.info('START train.py')
        args = vars(parse_args())
        logger.info('Load train df')
        train_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {train_df.shape}')
        train_df = prepare_categorical(train_df)
        train_df = clean_values(train_df)

        X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]
        X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES]
        y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]
        logger.info(f'X_offer {X_offer.shape}  y_offer {y_offer.shape}\tX_manual {X_manual.shape} y_manual {y_manual.shape}')
        model = BenchmarkModel(numerical_features=NUM_FEATURES, ohe_categorical_features=CATEGORICAL_OHE_FEATURES,
                                  ste_categorical_features=CATEGORICAL_STE_FEATURES, model_params=MODEL_PARAMS)
        logger.info('Fit model')
        model.fit(X_offer, y_offer, X_manual, y_manual)
        # logger.info('Save model')
        # model.save(args['mp'])

        predictions_offer = model.predict(X_offer)
        metrics = metrics_stat(y_offer.values, predictions_offer/(1+model.corr_coef)) # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
        logger.info(f'Metrics stat for training data with offers prices: {metrics}')

        predictions_manual = model.predict(X_manual)
        metrics = metrics_stat(y_manual.values, predictions_manual)
        logger.info(f'Metrics stat for training data with manual prices: {metrics}')

        # folds
        folds_list = []
        for i in range(N_CV_RUNS):
            # skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=(seed + i))
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=(SEED + i))
            folds_list.append([])
            for train_index, test_index in kf.split(train_df):
                folds_list[-1].append([train_index, test_index])

        X_train_stack = []
        for i in range(N_CV_RUNS):
            X_train_stack.append(train_df.copy())
        oof_preds, cv_models = get_oof(X_train_stack, train_df[TARGET], model, folds_list,
                n_cv_runs=N_CV_RUNS)

        predictions_offer_cv = np.squeeze(oof_preds[0][train_df.price_type == PriceTypeEnum.OFFER_PRICE])
        metrics = metrics_stat(y_offer.values, predictions_offer_cv / (
                    1 + model.corr_coef))
        logger.info(f'Metrics stat for training data with offers prices (cv): {metrics}')

        predictions_manual_cv = np.squeeze(oof_preds[0][train_df.price_type == PriceTypeEnum.MANUAL_PRICE])
        metrics = metrics_stat(y_manual.values, predictions_manual_cv)
        logger.info(f'Metrics stat for training data with manual prices (cv): {metrics}')

        logger.info('Save models (cv)')
        for i, c_id_models in enumerate(cv_models):
            for j, cv_model in enumerate(c_id_models):
                cv_model.save(f'{args["mp"]}_{i}_{j}')


    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise(e)
    logger.info('END train.py')