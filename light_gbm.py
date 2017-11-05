# coding:utf-8

import sys
import logging
import gc
import pickle
import gzip
import shutil
import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from utils import isrc_to_year
import config

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

logger.info('Loading data...')
data_path = '../input/'
if config.LOAD_META_DATA:
    train = pd.read_csv(config.TRAIN_DF_META_GZ, compression='gzip')
    test = pd.read_csv(config.TEST_DF_META_GZ, compression='gzip')
else:
    train = pd.read_csv(
        config.TRAIN_CSV_GZ, compression='gzip',
        dtype={
            'msno': 'category', 'source_system_tab': 'category',
            'source_screen_name': 'category', 'source_type': 'category',
            'target': np.uint8, 'song_id': 'category'
        }
    )
    test = pd.read_csv(
        config.TEST_CSV_GZ, compression='gzip',
        dtype={
            'msno': 'category', 'source_system_tab': 'category',
            'source_screen_name': 'category', 'source_type': 'category',
            'song_id': 'category'
        }
    )

    songs = pd.read_csv(
        config.SONGS_CSV_GZ, compression='gzip',
        dtype={
            'genre_ids': 'category', 'language': 'category',
            'artist_name': 'category', 'composer': 'category',
            'lyricist': 'category', 'song_id': 'category'
        }
    )

    members = pd.read_csv(
        config.MEMBERS_CSV_GZ, compression='gzip',
        dtype={
            'city': 'category', 'bd': np.uint8,
            'gender': 'category', 'registered_via': 'category'
        }
    )

    songs_extra = pd.read_csv(config.SONGS_EXTRA_INFO_CSV_GZ, compression='gzip')

    logger.info('Data preprocessing...')
    song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
    train = train.merge(songs[song_cols], on='song_id', how='left')
    test = test.merge(songs[song_cols], on='song_id', how='left')

    members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
    members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
    members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
    members = members.drop(['registration_init_time'], axis=1)

    songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
    songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')

    train = train.merge(songs_extra, on='song_id', how='left')
    test = test.merge(songs_extra, on='song_id', how='left')

    del members, songs
    gc.collect()

    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')

    train.to_csv(config.TRAIN_DF_META_GZ, index=False, compression='gzip', float_format='%.5f')
    test.to_csv(config.TEST_DF_META_GZ, index=False, compression='gzip', float_format='%.5f')

X = train.drop(['target'], axis=1)
y = train['target'].values

X_test = test.drop(['id'], axis=1)
ids = test['id'].values

del train, test
gc.collect()

d_train = lgb.Dataset(X, y)
watchlist = [d_train]

logger.info('Training LGBM model...')
params = dict()
params['learning_rate'] = 0.2
params['application'] = 'binary'
params['max_depth'] = 8
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'

if config.USE_PREDTRAINED_LGBM:
    model = pickle.load(gzip.open('{}.gz'.format(config.LGBM_MODEL), 'rb'))
else:
    model = lgb.train(
        params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, verbose_eval=5
    )
    pickle.dump(model, open(config.LGBM_MODEL, "wb"), protocol=3)
    with open(config.LGBM_MODEL, 'rb') as f_in, gzip.open('{}.gz'.format(config.LGBM_MODEL), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        os.remove(config.LGBM_MODEL)


logger.info('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv(config.LGBM_SUBMIT, compression='gzip', index=False, float_format='%.5f')
logger.info('Done!')
