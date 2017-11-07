# coding: utf-8
"""Расчёт рекомендаций

Пример запуска скрипта:

python3 fast_fm.py -m /home/alex/DataScience/MusicRecommender/input/
"""

import sys
import os
import logging
import pickle
import gzip
import shutil
import gc

from fastFM import sgd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy import io
from sklearn.metrics import roc_auc_score

from utils import user_sampling_from_df, reindex_df, create_encoders
import config


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class DataPreparator(object):
    def __init__(self, df, user_col, item_col, label_encoders, test_set_rate=0.8):
        self.data = df
        self.user_col = user_col
        self.item_col = item_col
        self.encoders = label_encoders
        self.test_set_rate = test_set_rate
        self.ui_matrix = None
        self.train_set = None
        self.valid_set = None
        self.shape = None
        self.items_shape = None
        
    def prepare_data(self):
        """data preparation step
        
        """
        self.train_test_split()
        self.reindex_df()
        if config.DUMP_TRAIN_DATA:
            self.dump()
    
    def train_test_split(self):
        self.valid_set, self.train_set = user_sampling_from_df(self.data, self.test_set_rate)
        logger.info(
            "train set={} rows, test set={} rows ({:.2f}  % from total)".
            format(
                self.train_set.shape[0],
                self.valid_set.shape[0],
                self.valid_set.shape[0] / self.train_set.shape[0])
        )

    def reindex_df(self):
        self.valid_set = reindex_df(self.valid_set)
        self.train_set = reindex_df(self.train_set)

    def dump(self):
        logger.info('Dumping matrices and DataFrames...')
        self.train_set.to_csv(config.TRAIN_DF_GZ, index=False, compression='gzip', float_format='%.5f')
        self.valid_set.to_csv(config.VALID_DF_GZ, index=False, compression='gzip', float_format='%.5f')
        pickle.dump(self.train_set.target.values, open(config.TRAIN_TARGET, "wb"), protocol=3)
        pickle.dump(self.valid_set.target.values, open(config.VALID_TARGET, "wb"), protocol=3)
        pickle.dump(self.encoders, open(config.ENCODERS, "wb"), protocol=3)
        logger.info("Dumping finished!")


class FeatureEncoder:
    def __init__(
            self, df, column_names, label_encoders,
            index_col='row_index', matrix_name=None, dump_name=None, is_dump=True
    ):
        self.df = df
        self.row_num = df.shape[0]
        self.shift = 0
        self.col_names = column_names
        self.encoders = label_encoders
        self.sparse_index = np.array([[], []]).astype(np.uint32).astype(np.uint32).reshape(0, 2)
        self.num_cols = label_encoders['num_cols']
        self.index_col = index_col
        self.feature_matrix = None
        self.dump_name = dump_name
        self.matrix_name = matrix_name
        self.is_dump = is_dump
    
    def build(self):
        for col_name in self.col_names:
            col_index = self.encoders[col_name].transform(self.df[col_name]) + self.shift
            feature_coded = np.array(list(zip(self.df[self.index_col], col_index))).astype(np.uint32)
            gc.collect()
            self.sparse_index = np.vstack([self.sparse_index, feature_coded]).astype(np.uint32)
            del feature_coded
            self.shift += (self.encoders[col_name].transform(self.df[col_name]).max()+1)
        logger.info("Creating sparse matrix shape = {}x{}".format(self.row_num, self.num_cols))
        self.feature_matrix = csr_matrix(
            (np.ones(self.sparse_index.shape[0]).astype(np.int8),
             (self.sparse_index[:, 0].astype(np.uint32), self.sparse_index[:, 1].astype(np.uint32))),
            shape=(self.row_num, self.num_cols)
        ).tocoo(copy=False).astype(np.int8)
        del self.sparse_index
        gc.collect()
        if self.is_dump:
            self.dump()

    def dump(self):
        logger.info("Dumping matrix {}...".format(self.dump_name))
        io.mmwrite(self.matrix_name, self.feature_matrix)
        with open(self.matrix_name, 'rb') as in_stream, gzip.open(self.dump_name, 'wb') as out_stream:
            shutil.copyfileobj(in_stream, out_stream)
            os.remove(self.matrix_name)


if __name__ == '__main__':
    test_df = None
    if len(os.listdir(config.WORKING_DIR)) == 0 or config.TRAIN_TEST_SPLIT:
        logger.info("Loading input train_csv, test_csv...")
        if config.LOAD_META_DATA:
            train_df = pd.read_csv(config.TRAIN_DF_META, compression='gzip')
            test_df = pd.read_csv(config.TEST_DF_META, compression='gzip')
            dtype_col = pd.read_pickle(config.META_DTYPES)
            train_df = train_df.astype(dtype=dtype_col)
            test_df = test_df.astype(dtype=dtype_col)
            train_df = train_df.drop(['song_length'], axis=1)
            test_df = test_df.drop(['song_length'], axis=1)
            y = pickle.load(open(config.TARGET_FULL_TRAIN_PKL, "rb"))
            train_df['target'] = y
            ids = pickle.load(open(config.IDS_FULL_TRAIN_PKL, "rb"))
            test_df['id'] = ids
            del y, ids
            gc.collect()
        else:
            train_df = pd.read_csv(config.TRAIN_CSV_GZ, compression='gzip')
            test_df = pd.read_csv(config.TEST_CSV_GZ, compression='gzip')
        # filling NaN with most frequent value
        train_df.fillna(value=train_df.mode().iloc[0], inplace=True)
        test_df.fillna(value=train_df.mode().iloc[0], inplace=True)
        col_names = np.unique([col for col in train_df.columns.append(test_df.columns) if col not in ('target', 'id')])
        pickle.dump(col_names, open(config.COL_NAMES, "wb"), protocol=3)

        logger.info(
            "Train-test intersection {:.2f} %"
            .format(
             100 * len((set(train_df.msno.unique()) & set(test_df.msno.unique()))) / len(set(train_df.msno.unique()))
            )
        )

        encoders = create_encoders(
            train_df, test_df, col_names
        )

        logger.info("Preparing raw data...")
        data_preparator = DataPreparator(
            train_df, 'msno', 'song_id', encoders, test_set_rate=0.2
        )
        del train_df
        gc.collect()
        data_preparator.prepare_data()

        logger.info("Feature columns for encoding: {} \nEncoding train data...".format(col_names))
        feature_coded_train = FeatureEncoder(
            data_preparator.train_set, col_names, encoders,
            matrix_name=config.TRAIN_MATRIX, dump_name=config.TRAIN_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
        )
        feature_coded_train.build()
        train_matrix = feature_coded_train.feature_matrix

        logger.info("Encoding valid data...")
        feature_coded_valid = FeatureEncoder(
            data_preparator.valid_set, col_names, encoders,
            matrix_name=config.VALID_MATRIX, dump_name=config.VALID_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
        )
        feature_coded_valid.build()
        valid_matrix = feature_coded_valid.feature_matrix

        train_target_values = data_preparator.train_set.target.values
        valid_target_values = data_preparator.valid_set.target.values

    else:
        logger.info('Загружаем test_df и train_df...')
        train_df = pd.read_csv(config.TRAIN_DF_GZ, compression='gzip')
        valid_df = pd.read_csv(config.VALID_DF_GZ, compression='gzip')
        logger.info('Загружаем матрицу test...')
        valid_matrix = io.mmread(gzip.open(config.VALID_MATRIX_GZ, 'rb'))
        logger.info('Загружаем матрицу train...')
        train_matrix = io.mmread(gzip.open(config.TRAIN_MATRIX_GZ, 'rb'))
        train_target_values = pickle.load(open(config.TRAIN_TARGET, 'rb'))
        valid_target_values = pickle.load(open(config.VALID_TARGET, 'rb'))
        col_names = pickle.load(open(config.COL_NAMES, 'rb'))
        encoders = pickle.load(open(config.ENCODERS, 'rb'))
        logger.info('Матрицы загружены!')

    logger.info('Train matrix shape={}, nnz={}, sparsity^(-1)={}'.format(
        train_matrix.shape, train_matrix.nnz,
        (train_matrix.shape[0]*train_matrix.shape[1])/train_matrix.nnz)
    )

    y_train = np.array([-1 if i == 0 else 1 for i in train_target_values])
    y_valid = np.array([-1 if i == 0 else 1 for i in valid_target_values])
    y_hat = None

    best_model = None
    if config.USE_PREDTRAINED_FM:
        logger.info('Загружаем обученную модель...')
        best_model = pickle.load(gzip.open('{}.gz'.format(config.FM_MODEL), 'rb'))
    else:
        rank = [2, 4, 8, 10, 12, 16, 24, 30, 32, 40, 50, 60, 64]

        best_score = 0
        for r in rank:
            model = sgd.FMClassification(
                n_iter=1000, init_stdev=0.1, l2_reg_w=0,
                l2_reg_V=0, rank=r, step_size=0.1, random_state=42
            )
            model.fit(train_matrix, y_train)

            y_hat = model.predict_proba(valid_matrix)
            score = roc_auc_score(valid_target_values, y_hat)
            best_model = model if score > best_score else best_model

            logger.info("r={}\tROC AUC = {}".format(r, score))
        print(y_valid, y_hat)

        logger.info('Сохраняем обученную модель (лучшую)...')
        pickle.dump(best_model, open(config.FM_MODEL, "wb"), protocol=3)
        with open(config.FM_MODEL, 'rb') as f_in, gzip.open('{}.gz'.format(config.FM_MODEL), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove(config.FM_MODEL)

    logger.info('Building test set...')
    test_df = pd.read_csv(config.TEST_CSV_GZ, compression='gzip') if test_df is None else test_df
    test_data = FeatureEncoder(
        test_df, col_names, encoders, index_col='id', is_dump=config.DUMP_TRAIN_DATA
    )
    test_data.build()

    logger.info('Fitting model on whole dataset (train+valid)...')
    union_data = vstack([train_matrix, valid_matrix], format='coo', dtype=np.int8)
    union_target = np.append(y_train, y_valid).astype(np.int8)
    del train_matrix, valid_matrix
    gc.collect()
    best_model.fit(union_data, union_target)
    logger.info('Making predictions...')
    y_predict = best_model.predict_proba(test_data.feature_matrix)
    result = pd.DataFrame({'id': test_df.id.values, 'target': y_predict})
    result.to_csv(config.FM_SUBMIT, index=False, sep=',', compression='gzip', float_format='%.5f')
    logger.info('Done!')
