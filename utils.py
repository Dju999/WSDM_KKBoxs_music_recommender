
# coding: utf-8
"""Вспомогательные функции

Работа с pandas.DataFrame и разреженными матрицами
"""
import gc

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix


def create_encoders(train_df, test_df, col_names):
    df = pd.concat([train_df, test_df])
    num_cols = 0
    encoders = {}
    for col in col_names:
        df[col].fillna(value=df[col].mode().iloc[0], inplace=True)
        df[col] = df[col].astype('category')
        encoders.update({
                '{}'.format(col): 
                LabelEncoder().fit( 
                    df[col]
                )
        }
        )
        num_cols += len(encoders['{}'.format(col)].classes_)
    encoders.update({'num_cols': num_cols})
    del df
    gc.collect()
    return encoders


def reindex_df(df):
    return df \
        .reset_index(drop=True) \
        .reset_index() \
        .rename(index=str, columns={'index': 'row_index'})


def matrix2df(matrix, index_col_name):
    matrix = matrix.tocsr()
    entries = []
    data_shape = matrix.shape[0]
    for i in range(data_shape):
        entries.append([i, np.array([j for j in matrix.getrow(i).nonzero()[1]])])
    code_df = pd.DataFrame(
        entries, columns=[index_col_name, '{}_code'.format(index_col_name)]
    )
    return code_df


def random_index_sample(total_rows, row_sample):
    sample_capacity = np.ceil(
        total_rows * row_sample
    ).astype(int)
    # row_num = range(total_rows)
    # return np.random.choice(row_num, sample_capacity, replace=False)
    return sample_capacity


def user_sampling_from_df(ui_df, user_sample):
    """Random sample of users
    
    filter sourse dataframe rows to select random subsample of users
    """
    num_rows = ui_df.shape[0]
    sample_capacity = random_index_sample(num_rows, user_sample)
    # preserve order of rows after sampling
    # ui_df = ui_df[ui_df[user_col_label].isin(random_index)]
    test = ui_df.sample(sample_capacity, random_state=42)
    inverted_index = test.index.difference(ui_df.index)
    return test, ui_df.loc[inverted_index]


def df2matrix(df, row_label, col_label, feedback_label, shape=None):
    """Convert
    
    """
    row_index = \
        df[row_label] if row_label is not None \
        else df.reset_index(drop=True).index.values
    col_index = df[col_label]
    feedback_values = np.ones(df.shape[0]) if feedback_label is None else df[feedback_label]
    if shape is None:
        shape = (row_index.max() + 1, col_index.max() + 1)
    ui_matrix = csr_matrix(
        (feedback_values, (row_index, col_index)),
        shape=shape
    ).astype(np.int8)
    return ui_matrix.tocsr()


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
