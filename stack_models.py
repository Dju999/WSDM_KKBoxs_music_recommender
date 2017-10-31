# coding:utf-8
import logging
import sys

import pandas as pd

import config

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    lgbm_submit = pd.read_csv(config.LGBM_SUBMIT, compression='gzip')
    fm_submit = pd.read_csv(config.FM_SUBMIT, compression='gzip')
    agg_submit = lgbm_submit.merge(fm_submit, on='id')
    result = pd.DataFrame({'id': agg_submit.id, 'target': agg_submit.target_x*0.9+agg_submit.target_y*0.1})
    result.to_csv(config.BAGGING_SUBMIT, compression='gzip', index=False, float_format='%.5f')
    logger.info('Bagging predictions saved!')
