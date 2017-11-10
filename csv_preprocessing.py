# coding:utf-8

import logging
import pickle

import pandas as pd
import numpy as np

import config
from utils import data_frame_normalize

if __name__ == '__main__':
    encoders = pickle.load(open(config.ENCODERS, 'rb'))

    """
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
    """

    songs = pd.read_csv(
        config.SONGS_CSV_GZ, compression='gzip'
    )
    songs['language'] = songs.language.fillna('-1')
    songs = songs.astype(dtype={
        'genre_ids': 'category', 'language': np.uint8,
        'artist_name': 'category', 'composer': 'category',
        'lyricist': 'category', 'song_id': 'category'
    })
    songs_flatten = data_frame_normalize(
        songs, index_col_name='song_id', sep='|',
        col_list=['genre_ids', 'composer', 'lyricist']
    )
    print(songs_flatten.head(100))

    """
    members = pd.read_csv(
        config.MEMBERS_CSV_GZ, compression='gzip'
    )
    members = members.astype(dtype={
        'city': 'category', 'bd': np.uint8,
        'gender': 'category', 'registered_via': 'category'
    })

    songs_extra = pd.read_csv(config.SONGS_EXTRA_INFO_CSV_GZ, compression='gzip')
    """
