
# coding: utf-8
"""Вспомогательные функции

Работа с pandas.DataFrame и разреженными матрицами
"""
import gc
import logging
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix

import config


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def data_frame_normalize(df, index_col_name, sep, col_list):
    """ Transform nested fields to flat

    :param df: source df
    :param index_col_name: primary DataFrame key
    :param sep: nested field separator
    :param col_list: columns for separation
    :return: result_df - flattened DataFrame
    """
    encoders = dict()
    invariant_labels = np.setdiff1d(df.columns.values, col_list+[index_col_name])
    for col_name in np.append(invariant_labels, index_col_name):
        df[col_name].fillna(df[col_name].mode().values[0], inplace=True)
        logger.info("Encodind col {}".format(col_name))
        encoders.update({col_name: LabelEncoder().fit(df[col_name])})
        df[col_name] = encoders[col_name].transform(df[col_name])
        df[col_name] = df[col_name].astype(np.uint32)
    result_df = df[np.append(index_col_name, invariant_labels)]
    print(result_df.head(10))

    for col_name in col_list:
        logger.info("Filling NA's in col {}".format(col_name))
        df[col_name].fillna(df[col_name].mode().values[0], inplace=True)
        df[col_name] = df[col_name].astype('category')
        df[col_name] = LabelEncoder().fit_transform(df[col_name])
        df[col_name] = df[col_name].astype(np.uint32)
        logger.info("Processing col {}".format(col_name))
        current_df = pd.DataFrame(
            np.vstack(df[[index_col_name, col_name]].apply(
                lambda row: [
                    (row[0], item)
                    for item in str(row[1]).split(sep)
                ],
                axis=1
            ).values),
            columns=[index_col_name, col_name]
        )
        current_df[index_col_name] = current_df[index_col_name].astype(np.uint32)
        logger.info('Encoding new col ...')
        print(current_df.head(10))
        current_df[col_name] = current_df[col_name].astype('category')
        current_df[col_name].fillna(current_df[col_name].mode().values[0], inplace=True)
        current_encoder = LabelEncoder().fit(current_df[col_name])
        current_df[col_name] = current_encoder.transform(current_df[col_name])
        current_df[col_name] = current_df[col_name].astype(np.uint32)
        print(result_df.head(1))
        result_df = result_df.merge(current_df, on=index_col_name)
        print(result_df.head(3))
        print(current_df.head(3))
        gc.collect()

    return result_df


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
    if config.LEAK_EXPLOITATION:
        test = ui_df.iloc[-2500000:]
    else:
        num_rows = ui_df.shape[0]
        sample_capacity = random_index_sample(num_rows, user_sample)
        test = ui_df.sample(sample_capacity, random_state=42)
    inverted_index = ui_df.index.difference(test.index)
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
