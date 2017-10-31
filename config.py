import os

PARENT_DIR = '/home/alex/Загрузки/input/'

WORKING_DIR = os.path.join(PARENT_DIR, 'output')

TRAIN_CSV_GZ = os.path.join(PARENT_DIR, 'train.csv.gz')
TEST_CSV_GZ = os.path.join(PARENT_DIR, 'test.csv.gz')
SONGS_CSV_GZ = os.path.join(PARENT_DIR, 'songs.csv.gz')
MEMBERS_CSV_GZ = os.path.join(PARENT_DIR, 'members.csv.gz')
SONGS_EXTRA_INFO_CSV_GZ = os.path.join(PARENT_DIR, 'song_extra_info.csv.gz')

TRAIN_MATRIX = os.path.join(WORKING_DIR, 'train.mtx')
VALID_MATRIX = os.path.join(WORKING_DIR, 'valid.mtx')

TRAIN_MATRIX_GZ = '{}.gz'.format(TRAIN_MATRIX)
VALID_MATRIX_GZ = '{}.gz'.format(VALID_MATRIX)

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

USE_PREDTRAINED_FM = True
USE_PREDTRAINED_LGBM = True
