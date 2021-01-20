from typing import Optional, List

import pandas as pd
import numpy as np


TEMP_NAN_VALUE = -100


def missing_data_imputation(df: pd.DataFrame,
                            missing_data_value='< 15',
                            imputed_value=1) -> pd.DataFrame:
    return df.replace(missing_data_value, imputed_value)


def convert_to_float(df: pd.DataFrame,
                     index_key: Optional[str] = None,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
    out_df = df.copy()
    if columns is not None:
        out_df = out_df[columns]

    if index_key is not None:
        out_df.set_index(index_key, inplace=True)

    out_df.replace(np.nan, TEMP_NAN_VALUE, inplace=True)
    out_df = out_df.astype('float')
    out_df.replace(TEMP_NAN_VALUE, np.nan, inplace=True)

    if index_key is not None:
        out_df.reset_index(inplace=True)

    return out_df


def convert_to_int(df: pd.DataFrame,
                   index_key: Optional[str] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    out_df = convert_to_float(df, index_key=index_key, columns=columns)

    if index_key is not None:
        out_df.set_index(index_key, inplace=True)

    out_df.replace(np.nan, TEMP_NAN_VALUE, inplace=True)
    out_df = out_df.astype('int')
    out_df.replace(TEMP_NAN_VALUE, np.nan, inplace=True)

    if index_key is not None:
        out_df.reset_index(inplace=True)

    return out_df


def filter_cities_with_missing_vaccinations_data(df: pd.DataFrame,
                                                 missing_data_thr: int = 1) -> pd.DataFrame:
    vaccination_df = df.copy()
    vaccination_df["drop_row"] = False
    for index, row in vaccination_df.iterrows():
        cnt = int(row["60-69"] == '< 15') + \
              int(row["70-79"] == '< 15') + \
              int(row["80-89"] == '< 15') + \
              int(row["90+"] == '< 15')
        if cnt > missing_data_thr:
            vaccination_df.loc[index, "drop_row"] = True
    vaccination_df = vaccination_df[vaccination_df.drop_row == False]
    vaccination_df = vaccination_df.drop("drop_row", 1)
    return vaccination_df
