import pandas as pd
import re
import numpy as np
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
    df_new['total_square'] = np.log(df_new['total_square'])

    return df_new


# city_population = pd.read_csv('../../data/city_population.csv')
# zarplaty = pd.read_excel('../../data/zarplaty.xlsx')
city_population = pd.read_csv('/media/alex/Plextor_1T/ML/2021_raifhack/data/city_population.csv')
zarplaty = pd.read_excel('/media/alex/Plextor_1T/ML/2021_raifhack/data/zarplaty.xlsx', engine='openpyxl')

def city_type(row):
    if row >= 1000000:
        return "1Million"
    elif (row < 1000000) & (row > 200000):
        return "Medium"
    elif (row <= 200000):
        return "Small"


def floor_type(row):
    if ('1' in str(row)) & (row != -1):
        return 1
    else:
        return 0

# ['age', 'city_type', 'zarplata', 'floor_type']
def add_features(df):
    df['age'] = round(2021 - df['reform_mean_year_building_500'])
    df.city = df.city.apply(lambda x: x.lower())

    city_population_clean = city_population.groupby('settlement').agg({'population': 'sum'}).reset_index()
    city_population_clean.columns = ['city', 'city_population']
    city_population_clean['city_population']
    city_population_clean.city = city_population_clean.city.apply(lambda x: x.lower())
    df = df.merge(city_population_clean, on='city', how='left')

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')

    df['city_type'] = df['city_population'].apply(lambda x: city_type(x))
    df.loc[df.city == 'москва', 'city_type'] = "Capital"
    df.loc[df.city == 'санкт-Петербург', 'city_type'] = "Capital"

    df = df.merge(zarplaty, on='region', how='left')
    df['zarplata'] = pd.to_numeric(df['zarplata'], downcast='unsigned')
    df['floor_type'] = df['floor'].apply(lambda x: floor_type(x))

    return df