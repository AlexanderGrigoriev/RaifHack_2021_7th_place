import pandas as pd
import re
from raif_hack.utils import UNKNOWN_VALUE

def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region','city','street','realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleansing values
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    df_new['floor'] = pd.to_numeric(df_new['floor'], errors='coerce')

    return df_new