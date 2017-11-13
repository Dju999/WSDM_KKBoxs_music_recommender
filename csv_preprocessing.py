# coding:utf-8

import logging
import sys
import gc
import pickle

import pandas as pd
import numpy as np

import config
from utils import data_frame_normalize, isrc_to_year, user_sampling_from_df


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def preprocess_csv():

    train = pd.read_csv(
        config.TRAIN_CSV_GZ, compression='gzip'
    )
    train = train.astype(dtype={
        'msno': 'category', 'source_system_tab': 'category',
        'source_screen_name': 'category', 'source_type': 'category',
        'target': np.uint8, 'song_id': 'category'
    })

    test = pd.read_csv(
        config.TEST_CSV_GZ, compression='gzip'
    )
    test = test.astype(dtype={
        'msno': 'category', 'source_system_tab': 'category',
        'source_screen_name': 'category', 'source_type': 'category',
        'song_id': 'category'
    })

    union_cols = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type']
    union_df = pd.concat([train[union_cols], test[union_cols]]).drop_duplicates()
    song_catalog = union_df[['song_id']].drop_duplicates().copy()
    msno_catalog = union_df[['msno']].drop_duplicates().copy()

    logger.info('Train user-item encoders...')
    data_frame_normalize(
        union_df, index_col_name='msno', sep='|',
        col_list=[]
    )
    del union_df
    gc.collect()

    valid, train = user_sampling_from_df(train, config.TEST_SET_SAMPLE)

    songs = pd.read_csv(
        config.SONGS_CSV_GZ, compression='gzip'
    )
    songs[['language', 'song_length']] = songs[['language', 'song_length']].fillna('-1')
    songs = songs[['song_id', 'song_length', 'genre_ids', 'language']]
    songs = songs.astype(dtype={
        'genre_ids': 'category', 'language': np.uint8,
        'song_id': 'category', 'song_length': np.uint16
    })
    songs_cols = songs.columns
    num_rows = songs.shape[0]
    songs = songs.merge(song_catalog, on='song_id', how='inner')[songs_cols].drop_duplicates()
    logger.info('Songs delta filtering: {} before, {} after'.format(num_rows, songs.shape[0]))

    songs_normalized = data_frame_normalize(
        songs, index_col_name='song_id', sep='|',
        col_list=['genre_ids']
    )
    songs_normalized.to_csv(
        config.ENCODED_SONGS_CSV_GZ, index=False, float_format='%.5f', encoding='utf-8', compression='gzip'
    )

    members = pd.read_csv(
        config.MEMBERS_CSV_GZ, compression='gzip'
    )
    members = members.astype(dtype={
        'city': 'category', 'bd': np.uint8,
        'gender': 'category', 'registered_via': 'category'
    })
    members_cols = members.columns
    num_rows = members.shape[0]
    members = members.merge(msno_catalog, on='msno', how='inner')[members_cols].drop_duplicates()

    members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4])).astype(np.uint16)
    members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)

    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4])).astype(np.uint16)
    members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)
    members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8])).astype(np.uint8)
    members = members.drop(['registration_init_time'], axis=1)

    logger.info('Members delta filtering: {} before, {} after'.format(num_rows, members.shape[0]))
    members_normalized = data_frame_normalize(
        members, index_col_name='msno', sep='|',
        col_list=['gender']
    )
    members_normalized.to_csv(
        config.ENCODED_MEMBERS_CSV_GZ, index=False, float_format='%.5f', encoding='utf-8', compression='gzip'
    )

    songs_extra = pd.read_csv(config.SONGS_EXTRA_INFO_CSV_GZ, compression='gzip')
    songs_extra['song_id'] = songs_extra['song_id'].astype('category')

    songs_extra_cols = songs_extra.columns
    songs_extra = songs_extra.merge(
        song_catalog[['song_id']], on='song_id', how='inner'
    )[songs_extra_cols].drop_duplicates()

    songs_extra.fillna(value=songs_extra.mode().iloc[0], inplace=True)

    songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
    songs_extra['song_year'] = songs_extra['song_year'].astype(np.uint16)
    songs_extra['song_id'] = songs_extra['song_id'].astype('category')
    songs_extra['song_id'] = config.encoders['song_id'].transform(songs_extra['song_id'])
    songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)

    context_columns = ['source_system_tab', 'source_screen_name', 'source_type']

    train_normalized = data_frame_normalize(
        train, index_col_name='msno', sep='|',
        col_list=[]
    )

    valid_normalized = data_frame_normalize(
        valid, index_col_name='msno', sep='|',
        col_list=[]
    )

    test_normalized = data_frame_normalize(
        test, index_col_name='msno', sep='|',
        col_list=[]
    )

    logger.info('Data encoding finished! Adding meta-data...')
    pickle.dump(config.encoders, open(config.ENCODERS, "wb"), protocol=3)

    train_normalized = train_normalized.merge(songs_normalized, on='song_id', how='inner')
    valid_normalized = valid_normalized.merge(songs_normalized, on='song_id', how='inner')
    test_normalized = test_normalized.merge(songs_normalized, on='song_id', how='inner')

    train_normalized = train_normalized.merge(members_normalized, on='msno')
    valid_normalized = valid_normalized.merge(members_normalized, on='msno')
    test_normalized = test_normalized.merge(members_normalized, on='msno')

    train_normalized = train_normalized.merge(songs_extra, on='song_id', how='left')
    valid_normalized = valid_normalized.merge(songs_extra, on='song_id', how='left')
    test_normalized = test_normalized.merge(songs_extra, on='song_id', how='left')

    test_normalized.to_csv(
        config.ENCODED_TEST_CSV_GZ, index=False, float_format='%.5f', encoding='utf-8', compression='gzip'
    )

    valid_normalized.to_csv(
        config.ENCODED_VALID_CSV_GZ, index=False, float_format='%.5f', encoding='utf-8', compression='gzip'
    )

    train_normalized.to_csv(
        config.ENCODED_TRAIN_CSV_GZ, index=False, float_format='%.5f', encoding='utf-8', compression='gzip'
    )

    print('train set {}'.format(train_normalized.shape))
    print('train set {}'.format(valid_normalized.shape))
    print('test set {}'.format(test_normalized.shape))
    print(valid_normalized.head(3))
    del members, songs, songs_extra, members_normalized, songs_normalized, train, test
    gc.collect()
    pickle.dump(config.encoders, open(config.ENCODERS, "wb"), protocol=3)

if __name__ == '__main__':
    preprocess_csv()
