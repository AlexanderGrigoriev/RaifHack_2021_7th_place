import argparse
import logging.config
import numpy as np
import pandas as pd
import glob
from raif_hack.features import prepare_categorical, clean_values
from traceback import format_exc


from raif_hack.model import BenchmarkModel
from raif_hack.settings import LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES, \
    CATEGORICAL_STE_FEATURES

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
    Бенчмарк для хакатона по предсказанию стоимости коммерческой недвижимости от "Райффайзенбанк"
    Скрипт для предсказания модели
     
     Примеры:
        1) с poetry - poetry run python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
        2) без poetry - python3 predict.py --test_data /path/to/test/data --model_path /path/to/model --output /path/to/output/file.csv.gzip
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_data", "-d", type=str, dest="d", required=True, help="Путь до отложенной выборки")
    parser.add_argument("--model_path", "-mp", type=str, dest="mp", required=True,
                        help="Пусть до сериализованной ML модели")
    parser.add_argument("--output", "-o", type=str, dest="o", required=True, help="Путь до выходного файла")

    return parser.parse_args()

if __name__ == "__main__":

    try:
        logger.info('START predict.py')
        args = vars(parse_args())
        logger.info('Load test df')
        test_df = pd.read_csv(args['d'])
        logger.info(f'Input shape: {test_df.shape}')
        test_df = prepare_categorical(test_df)
        test_df = clean_values(test_df)

        logger.info('Load model')
        model_fnames = glob.glob(f'{args["mp"]}*')
        models = [BenchmarkModel.load(model_fname) for model_fname in model_fnames]
        logger.info('Predict')
        models_preds = np.zeros((len(models), test_df.shape[0]))
        for i in range(len(models)):
            models_preds[i] = models[i].predict(test_df[NUM_FEATURES+CATEGORICAL_OHE_FEATURES+CATEGORICAL_STE_FEATURES])
        mean_preds = np.mean(models_preds, axis=0)
        test_df['per_square_meter_price'] = mean_preds
        logger.info('Save results (cv)')
        test_df['per_square_meter_price'] *= .98
        test_df[['id','per_square_meter_price']].to_csv(args['o'], index=False)
    except Exception as e:
        err = format_exc()
        logger.error(err)
        raise (e)

    logger.info('END predict.py')