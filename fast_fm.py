# coding: utf-8
"""Расчёт рекомендаций

Пример запуска скрипта:

python3 fast_fm.py -m /home/alex/DataScience/MusicRecommender/input/
"""

from subprocess import check_output
import sys
import os
import logging
import pickle
import argparse

from fastFM import sgd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
from scipy import io
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import user_sampling_from_df, reindex_df, create_encoders

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class DataPreparator(object):
    def __init__(self, df, user_col, item_col, encoders, test_set_rate=0.8):
        self.data = df
        self.user_col = user_col
        self.item_col = item_col
        self.encoders = encoders
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
    
    def train_test_split(self):
        self.valid_set = user_sampling_from_df(self.data, self.user_col, self.test_set_rate)
        self.train_set = self.data[
            np.logical_not(
                self.data.index.isin(self.valid_set.index)
            )
        ]
        logger.info("train set={} rows, test set={} rows ({:.2f}  % from total)".\
              format(self.train_set.shape[0], self.valid_set.shape[0], self.valid_set.shape[0] / self.train_set.shape[0]))

    def reindex_df(self):
        self.valid_set = reindex_df(self.valid_set)
        self.train_set = reindex_df(self.train_set)


class FeatureEncoder:
    def __init__(self, df, col_names, encoders, num_cols, amplifying_coef=1, index_col = 'row_index'):
        self.df = df
        self.row_num = df.shape[0]
        self.shift = 0
        self.col_names = col_names
        self.encoders = encoders
        self.sparse_index = np.array([[],[]]).reshape(0,2)
        self.num_cols = num_cols
        self.amplifying_coef = amplifying_coef
        self.index_col = index_col
    
    def build(self):
        for col_name in self.col_names:
            # fill missing values
            non_code_labels = self.df[col_name].fillna('Unknown')
            col_index = self.encoders[col_name].transform(non_code_labels) + self.shift
            feature_coded = np.array(list(zip(self.df[self.index_col], col_index)))
            self.sparse_index  = np.vstack([self.sparse_index, feature_coded])
            self.shift += (self.encoders[col_name].transform(non_code_labels).max()+1)
        self.feature_matrix = csr_matrix(
            (np.ones(self.sparse_index.shape[0])*self.amplifying_coef, 
             (self.sparse_index[:,0], self.sparse_index[:,1])),
             shape = (self.row_num, self.num_cols)
        )


AMP_COEFF = 1.0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # разбираем параметры командной строки
    parser.add_argument('-m', '--matrix_directory', dest='matrix_directory', help='Директория с предрасчитанными матрицами')
    args = parser.parse_args()

    test_df = None
    if True:#len(os.listdir(args.matrix_directory))==0:
        logger.info("Loading input train_csv, test_csv...")
        train_df = pd.read_csv('../input/train.csv')
        test_df = pd.read_csv('../input/test.csv')

        col_names = [col for col in train_df.columns.append(test_df.columns) if col not in ('target','id')]

        logger.info("Train-test intersection {:.2f} %".format(100*len((set(train_df.msno.unique())&set(test_df.msno.unique())))/ len(set(train_df.msno.unique()))))

        encoders, num_cols = create_encoders(
            train_df, test_df,
            col_names
        )

        data_preparator = DataPreparator(
            train_df, 
            'msno', 
            'song_id', 
            encoders, 
            test_set_rate=0.2
        )

        data_preparator.prepare_data()

        logger.info("Feature columns for encoding: {}".format(col_names))

        logger.info("Encoding train data...")
        feature_coded_train = FeatureEncoder(
            data_preparator.train_set, 
            col_names,
            encoders,
            num_cols,
            amplifying_coef = AMP_COEFF
        )
        feature_coded_train.build()
        logger.info("Dumping train matrix...")
        io.mmwrite('../input/train.mtx', feature_coded_train.feature_matrix)
        train_matrix = feature_coded_train.feature_matrix

        logger.info("Encoding valid data...")
        feature_coded_valid = FeatureEncoder(
            data_preparator.valid_set, 
            col_names,
            encoders,
            num_cols,
            amplifying_coef = AMP_COEFF
        )
        feature_coded_valid.build()
        logger.info("Dumping valid matrix...")
        io.mmwrite('../input/valid.mtx', feature_coded_valid.feature_matrix)
        valid_matrix = feature_coded_valid.feature_matrix

        logger.info("Dumping matrices...")
        pickle.dump(data_preparator.train_set.target.values, open('../input/train_target.pkl', "wb"), protocol=3)
        data_preparator.train_set.to_csv('../input/train_df.csv', index='False')
        data_preparator.valid_set.to_csv('../input/valid_df.csv', index='False')
        pickle.dump(data_preparator.valid_set.target.values, open('../input/valid_target.pkl', "wb"), protocol=3)
        pickle.dump(col_names, open('../input/col_names.pkl', "wb"), protocol=3)
        train_target_values = data_preparator.train_set.target.values
        valid_target_values = data_preparator.valid_set.target.values

    else:
        logger.info('Загружаем test_df и train_df...')
        train_df = pd.read_csv('../input/train_df.csv', index='False')
        valid_df = pd.read_csv('../input/valid_df.csv', index='False')
        logger.info('Загружаем матрицу test...')
        valid_matrix = io.mmread('../input/valid.mtx')
        logger.info('Загружаем матрицу train...')
        train_matrix = io.mmread('../input/train.mtx')
        train_target_values = pickle.load(open('../input/train_target.pkl', 'rb'))
        valid_target_values = pickle.load(open('../input/valid_target.pkl', 'rb'))
        col_names = pickle.load(open('../input/col_names.pkl', 'rb'))
        logger.info('Матрицы загружены!')

    test_df = pd.read_csv('../input/test.csv') if test_df is None else test_df

    logger.info('Train matrix shape={}, nnz={}, sparsity^(-1)={}'.format(
        train_matrix.shape, 
        train_matrix.nnz,
        (train_matrix.shape[0]*train_matrix.shape[1])/train_matrix.nnz)
    )

    y_train = np.array([-1 if i==0 else 1 for i in train_target_values])
    y_train.min()

    rank = [2, 4, 8, 10, 12, 16, 24, 30, 32, 40, 50, 60, 64, 128]

    best_score = 0
    best_model = None
    for r in rank:
        model = sgd.FMClassification(
            n_iter=1000, init_stdev=0.1, l2_reg_w=0,
            l2_reg_V=0, rank=r, step_size=0.1, random_state=42
        )
        model.fit(train_matrix, y_train)

        y_hat = model.predict_proba(valid_matrix)
        score = roc_auc_score(valid_target_values, y_hat)
        best_model = model if score > best_score else best_model

        logger.info("r={}\tROC AUC = {}".format(r, score ))


    logger.info('Сохраняем обученную модель...')
    pickle.dump(best_model, open('../input/best_model.pkl', "wb"), protocol=3)

    logger.info('Building test set...')
    test_data = FeatureEncoder(
        test_df, 
        col_names,
        encoders,
        num_cols,
        amplifying_coef = AMP_COEFF,
        index_col = 'id'
    )

    test_data.build()
    logger.info('Making predictions...')
    y_predict = best_model.predict_proba(test_data.feature_matrix)
    result = pd.DataFrame({'id': test_df.id.values,'target': y_predict})
    result.to_csv('../input/submit.csv', index=False, sep=',', compression = 'gzip', float_format = '%.5f')
    logger.info('Done!')
