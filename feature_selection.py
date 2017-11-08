# coding:utf-8

import gc

import pandas as pd

import config


logger.info('Loading preprocessed data...')
train_df = pd.read_csv(config.TRAIN_DF_META, compression='gzip')
dtype_col = pd.read_pickle(config.META_DTYPES)
train_df = train_df.astype(dtype=dtype_col)
train_df = train_df.drop(['song_length'], axis=1)
y = pickle.load(open(config.TARGET_FULL_TRAIN_PKL, "rb"))
train_df['target'] = y
ids = pickle.load(open(config.IDS_FULL_TRAIN_PKL, "rb"))
test_df['id'] = ids
del y, ids
gc.collect()

col_names = np.unique([col for col in train_df.columns.append if col not in ('target', 'id')])

encoders = create_encoders(
    train_df, train_df, col_names
)

# Тут запускать алгоритм расчёта фичей

logger.info("Preparing raw data...")
data_preparator = DataPreparator(
    train_df, 'msno', 'song_id', encoders, test_set_rate=0.2
)
del train_df
gc.collect()
data_preparator.prepare_data()