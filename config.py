import os
import numpy as np

PARENT_DIR = '/home/alex/input'

WORKING_DIR = os.path.join(PARENT_DIR, 'output')

TRAIN_CSV_GZ = os.path.join(PARENT_DIR, 'train.csv.gz')
TEST_CSV_GZ = os.path.join(PARENT_DIR, 'test.csv.gz')
SONGS_CSV_GZ = os.path.join(PARENT_DIR, 'songs.csv.gz')
MEMBERS_CSV_GZ = os.path.join(PARENT_DIR, 'members.csv.gz')
SONGS_EXTRA_INFO_CSV_GZ = os.path.join(PARENT_DIR, 'song_extra_info.csv.gz')
#
ENCODED_TRAIN_CSV_GZ = os.path.join(WORKING_DIR, 'train.csv.gz')
ENCODED_VALID_CSV_GZ = os.path.join(WORKING_DIR, 'valid.csv.gz')
ENCODED_TEST_CSV_GZ = os.path.join(WORKING_DIR, 'test.csv.gz')
ENCODED_SONGS_CSV_GZ = os.path.join(WORKING_DIR, 'songs.csv.gz')
ENCODED_MEMBERS_CSV_GZ = os.path.join(WORKING_DIR, 'members.csv.gz')
ENCODED_SONGS_EXTRA_INFO_CSV_GZ = os.path.join(WORKING_DIR, 'song_extra_info.csv.gz')

TRAIN_DF_META = os.path.join(WORKING_DIR, 'train_meta.csv.gz')
TEST_DF_META = os.path.join(WORKING_DIR, 'test_meta.csv.gz')

TRAIN_DF_META_FLATTEN = os.path.join(WORKING_DIR, 'train_meta_flatten.csv.gz')
TEST_DF_META_FLATTEN = os.path.join(WORKING_DIR, 'test_meta_flatten.csv.gz')

TARGET_FULL_TRAIN_PKL = os.path.join(WORKING_DIR, 'target_full_train.pkl')
IDS_FULL_TRAIN_PKL = os.path.join(WORKING_DIR, 'ids_full_train.pkl')

META_DTYPES = os.path.join(WORKING_DIR, 'meta_dtypes.pkl')

TRAIN_MATRIX = os.path.join(WORKING_DIR, 'train.mtx')
VALID_MATRIX = os.path.join(WORKING_DIR, 'valid.mtx')
TRAIN_MATRIX_META = os.path.join(WORKING_DIR, 'train_meta.mtx')
VALID_MATRIX_META = os.path.join(WORKING_DIR, 'valid_meta.mtx')

TRAIN_MATRIX_GZ = '{}.gz'.format(TRAIN_MATRIX)
VALID_MATRIX_GZ = '{}.gz'.format(VALID_MATRIX)
TRAIN_MATRIX_META_GZ = '{}.gz'.format(TRAIN_MATRIX_META)
VALID_MATRIX_META_GZ = '{}.gz'.format(VALID_MATRIX_META)

TRAIN_DF = os.path.join(WORKING_DIR, 'train_df.csv')
VALID_DF = os.path.join(WORKING_DIR, 'valid_df.csv')

TRAIN_TARGET = os.path.join(WORKING_DIR, 'train_target.pkl')
VALID_TARGET = os.path.join(WORKING_DIR, 'valid_target.pkl')

TRAIN_DF_GZ = os.path.join(WORKING_DIR, 'train_df.csv.gz')
VALID_DF_GZ = os.path.join(WORKING_DIR, 'valid_df.csv.gz')

FM_MODEL = os.path.join(WORKING_DIR, 'fm_model.pkl')
LGBM_MODEL = os.path.join(WORKING_DIR, 'lgbm_model.pkl')
COL_NAMES = os.path.join(WORKING_DIR, 'col_names.pkl')

ENCODERS = os.path.join(WORKING_DIR, 'encoders.pkl')

FM_SUBMIT = os.path.join(WORKING_DIR, 'fm_submission.csv.gz')
LGBM_SUBMIT = os.path.join(WORKING_DIR, 'lgbm_submission.csv.gz')
BAGGING_SUBMIT = os.path.join(WORKING_DIR, 'bagging_submission.csv.gz')


FEATURE_SELECTION_LOG = os.path.join(WORKING_DIR, 'feature_selection_log.csv')

USE_PREDTRAINED_FM = False
USE_PREDTRAINED_LGBM = False
TRAIN_TEST_SPLIT = True
LOAD_META_DATA = False
DUMP_TRAIN_DATA = False

LEAK_EXPLOITATION = True

(TRAIN_MATRIX_GZ, VALID_MATRIX_GZ) = \
    (TRAIN_MATRIX_GZ, VALID_MATRIX_GZ) \
    if LOAD_META_DATA \
    else (TRAIN_MATRIX_META_GZ, VALID_MATRIX_META_GZ)

(TRAIN_MATRIX, VALID_MATRIX) = \
    (TRAIN_MATRIX, VALID_MATRIX) \
    if LOAD_META_DATA \
    else (TRAIN_MATRIX_META, VALID_MATRIX_META)

USE_OFFLINE_PARAMETERS_ESTIMATION = False
OFFLINE_PARAMS = {
    'feature_space': [
        'msno', 'song_id', 'source_screen_name', 'source_system_tab', 'expiration_year', 'gender', 'expiration_date'
    ],
    'rank': 25,
}

# encoders for categorical data
encoders = dict()

TEST_SET_SAMPLE = 0.2

train_dtypes = {
        'row_index': np.int32,
        'song_id': np.int32,
        'source_screen_name': np.int8,
        'source_system_tab': np.int8,
        'source_type': np.int8,
        'target': np.int8,
        'msno': np.int32,
        'language': np.int32,
        'genre_ids': np.int32,
        'bd': np.int32,
        'city': np.int8,
        'expiration_date': np.int8,
        'expiration_month': np.int8,
        'expiration_year': np.int8,
        'registered_via': np.int8,
        'registration_date': np.int8,
        'registration_month': np.int8,
        'registration_year': np.int8,
        'gender': np.int8,
        'song_year': np.float64
}
