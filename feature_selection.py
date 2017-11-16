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

    rank = [2, 8, 10, 16, 24, 30, 40, 50, 60]
    best_rank_list = []
    best_features_list = []
    best_score_list = []
    initial_col_list = np.array(['msno', 'song_id', 'row_index'])
    candidates_feature_list = (np.setdiff1d(col_names, initial_col_list))

    # Тут запускать алгоритм расчёта фичей

    for r in rank:
        model = sgd.FMClassification(
            n_iter=1000, init_stdev=0.1, l2_reg_w=0,
            l2_reg_V=0, rank=r, step_size=0.1, random_state=42
        )
        best_features_list = initial_col_list.copy()
        model.fit(
            FeatureEncoder(
                train_df, initial_col_list, label_encoders=encoders
            ).build(transform=False, verbose=False).feature_matrix, y_train
        )
        y_hat = model.predict_proba(
            FeatureEncoder(
                valid_df, initial_col_list, label_encoders=encoders
            ).build(transform=False, verbose=False).feature_matrix
        )
        global_best_score = roc_auc_score(valid_target_values, y_hat)
        print("Minimal score = {}".format(global_best_score))
        current_candidates = candidates_feature_list.copy()
        feature_weight = dict()
        while current_candidates.shape[0] > 0:
            current_best_score = 0
            current_best_feature = None
            # перебираем фичи, ищем оптимальную фичу для добавления
            for feature_candidate in current_candidates:
                feature_space = np.append(best_features_list, feature_candidate)
                feature_coded_train = FeatureEncoder(
                    train_df, feature_space, label_encoders=encoders,
                    matrix_name=config.TRAIN_MATRIX, dump_name=config.TRAIN_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
                )
                feature_coded_train.build(transform=False, verbose=False)
                train_matrix = feature_coded_train.feature_matrix

                feature_coded_valid = FeatureEncoder(
                    valid_df, feature_space, encoders,
                    matrix_name=config.VALID_MATRIX, dump_name=config.VALID_MATRIX_GZ, is_dump=config.DUMP_TRAIN_DATA
                )
                feature_coded_valid.build(transform=False, verbose=False)
                valid_matrix = feature_coded_valid.feature_matrix
                # вычисляем скор модели для этой фичи
                model.fit(train_matrix, y_train)
                y_hat = model.predict_proba(valid_matrix)
                score = roc_auc_score(valid_target_values, y_hat)
                if current_candidates.shape[0] == candidates_feature_list.shape[0]:
                    # at the first iteration evaluate candidate's gain
                    feature_weight.update({feature_candidate: (score - current_best_score)})
                if score > current_best_score:
                    current_best_score = score
                    current_best_feature = feature_candidate
            if current_best_score > global_best_score:
                # добавляем лучшую фичу
                best_features_list = np.append(best_features_list, current_best_feature)
                # удаляем эту фичу из списка кандидатов
                current_candidates = np.setdiff1d(col_names, best_features_list)
                # ранжируем кандидатов по их вкладу в ROC_AUC
                candidates_weight = {col_name: feature_weight[col_name] for col_name in current_candidates}
                current_candidates = np.array(
                        sorted(
                            candidates_weight
                            , key=candidates_weight.__getitem__, reverse=True)
                    )
                print("Веса фичей {}".format(candidates_weight))
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
