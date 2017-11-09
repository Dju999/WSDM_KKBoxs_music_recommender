# coding:utf-8

import gc
import pickle
import logging
import sys

import pandas as pd
import numpy as np
from fastFM import sgd
from sklearn.metrics import roc_auc_score

import config
from utils import create_encoders
from fast_fm import DataPreparator, FeatureEncoder


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    logger.info('Loading preprocessed data...')
    train_df = pd.read_csv(config.TRAIN_DF_META, compression='gzip')
    dtype_col = pd.read_pickle(config.META_DTYPES)
    train_df = train_df.astype(dtype=dtype_col)
    train_df = train_df.drop(['song_length'], axis=1)
    y = pickle.load(open(config.TARGET_FULL_TRAIN_PKL, "rb"))
    train_df['target'] = y
    del y
    gc.collect()

    col_names = np.unique([col for col in train_df.columns if col not in ('target', 'id')])

    encoders = create_encoders(
        train_df, train_df, col_names
    )

    logger.info("Preparing raw data...")
    data_preparator = DataPreparator(
        train_df, 'msno', 'song_id', encoders, test_set_rate=0.2
    )
    del train_df
    gc.collect()
    data_preparator.prepare_data()

    train_target_values = data_preparator.train_set.target.values
    valid_target_values = data_preparator.valid_set.target.values
    y_train = np.array([-1 if i == 0 else 1 for i in train_target_values])
    y_valid = np.array([-1 if i == 0 else 1 for i in valid_target_values])
    y_hat = None

    rank = [2, 8, 10, 16, 24, 30, 40, 50, 60]
    best_rank_list = []
    best_features_list = []
    best_score_list = []
    initial_col_list = np.array(['msno', 'song_id'])
    candidates_feature_list = (np.setdiff1d(col_names, initial_col_list))

    # Тут запускать алгоритм расчёта фичей

    for r in rank:
        model = sgd.FMClassification(
            n_iter=1000, init_stdev=0.1, l2_reg_w=0,
            l2_reg_V=0, rank=r, step_size=0.1, random_state=42
        )
        best_features_list = initial_col_list.copy()
        global_best_score = 0
        current_candidates = candidates_feature_list.copy()
        while current_candidates.shape[0] > 0:
            current_best_score = 0
            current_best_feature = None
            # перебираем фичи, ищем оптимальную фичу для добавления
            for feature_candidate in current_candidates:
                feature_space = np.append(best_features_list, feature_candidate)
                feature_coded_train = FeatureEncoder(
                    data_preparator.train_set, feature_space, encoders,
                    matrix_name=config.TRAIN_MATRIX, dump_name=config.TRAIN_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
                )
                feature_coded_train.build(verbose=False)
                train_matrix = feature_coded_train.feature_matrix

                feature_coded_valid = FeatureEncoder(
                    data_preparator.valid_set, feature_space, encoders,
                    matrix_name=config.VALID_MATRIX, dump_name=config.VALID_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
                )
                feature_coded_valid.build(verbose=False)
                valid_matrix = feature_coded_valid.feature_matrix
                # вычисляем скор модели для этой фичи
                model.fit(train_matrix, y_train)
                y_hat = model.predict_proba(valid_matrix)
                score = roc_auc_score(valid_target_values, y_hat)
                if score > current_best_score:
                    current_best_score = score
                    current_best_feature = feature_candidate
            if current_best_score > global_best_score and current_best_score-global_best_score > 10**(-5):
                # добавляем лучшую фичу
                best_features_list = np.append(best_features_list, current_best_feature)
                # удаляем эту фичу из списка кандидатов
                current_candidates = np.setdiff1d(col_names, best_features_list)
                # логируем информацию
                logger.info(
                    "r={}\tROC AUC={:.4f}\tgain={:.4f}\tfeature_to_add={}"
                    .format(r, current_best_score, current_best_score - global_best_score, current_best_feature)
                )
                # обновляем скор
                global_best_score = current_best_score
            else:
                logger.info('Набрали фичей: {} для ранга {} . Скор {}'
                            .format(','.join(best_features_list), r, global_best_score)
                )
                current_candidates = np.array([])
                break
        logger.info("Закончили отбор фичей")
