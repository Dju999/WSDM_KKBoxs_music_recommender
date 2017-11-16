# coding:utf-8

import pickle
import logging
import sys

import numpy as np
import pandas as pd
import lightfm

import config
from utils import create_encoders
from fast_fm import FeatureEncoder

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    logger.info('Loading preprocessed data...')
    train_df = pd.read_csv(config.ENCODED_TRAIN_CSV_GZ, compression='gzip')
    valid_df = pd.read_csv(config.ENCODED_VALID_CSV_GZ, compression='gzip')
    train_df = train_df.drop(['song_length'], axis=1)
    valid_df = valid_df.drop(['song_length'], axis=1)
    train_df = train_df.astype(config.train_dtypes)
    valid_df = valid_df.astype(config.train_dtypes)
    valid_df['song_year'] = valid_df['song_year'].fillna(valid_df['song_year'].mode().values[0]).astype(np.uint8)
    train_df['song_year'] = train_df['song_year'].fillna(train_df['song_year'].mode().values[0]).astype(np.uint8)
    print(train_df.head(1))

    encoders = pickle.load(open(config.ENCODERS, "rb"))
    print("Encoded columns: {}".format(encoders.keys()))

    col_names = np.unique([col for col in train_df.columns if col not in ('target', 'id')])

    train_target_values = train_df.target.values
    valid_target_values = valid_df.target.values
    y_train = np.array([-1 if i == 0 else 1 for i in train_target_values])
    y_valid = np.array([-1 if i == 0 else 1 for i in valid_target_values])
    y_hat = None

    initial_col_list = np.array(['msno', 'song_id', 'row_index'])
    candidates_feature_list = (np.setdiff1d(col_names, initial_col_list))

    print("Encode train data frame ".format(encoders.keys()))
    train_matrix = FeatureEncoder(
        train_df, initial_col_list, label_encoders=encoders
    ).build(transform=False, verbose=False).feature_matrix

    print("Encode valid data frame ".format(encoders.keys()))
    train_matrix = FeatureEncoder(
        train_df, initial_col_list, label_encoders=encoders
    ).build(transform=False, verbose=False).feature_matrix

    valid_matrix = FeatureEncoder(
        valid_df, initial_col_list, label_encoders=encoders
    ).build(transform=False, verbose=False).feature_matrix

    # ищем пользователей в трейне и тесте
    common_users = np.intersect1d(train_matrix.nonzero()[0], valid_matrix.nonzero()[0])
    # находим маску - индексы, которые им соответствуют

    common_user_mask_train = np.in1d(train_matrix.nonzero()[0], common_users)
    train_content = train_matrix.nonzero()[1][common_user_mask_train]
    common_user_mask_valid = np.in1d(train_matrix.nonzero()[0], common_users)
    valid_content = train_matrix.nonzero()[1][common_user_mask_valid]
    print(train_content)


