import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 不显示tensorflow调试信息
import logging
import time
import json

import numpy as np

from usad.utils import get_data, ConfigHandler, merge_data_to_csv, paint, pot_detect
from usad.model import USAD

def main():
    logging.basicConfig(
            level='INFO',
            format='[%(asctime)s [%(levelname)s]] %(message)s')
    
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size,
                 train_start=config.train_start, test_start=config.test_start, do_preprocess=True,
                 prefix='../Compared/OmniAnomaly/processed')

    model = USAD(x_dims=config.x_dims, max_epochs=config.max_epochs,
                 batch_size=config.batch_size, z_dims=config.z_dims,
                 window_size=config.window_size,
                 valid_step_frep=config.valid_step_freq,
                 )

    model_metrics = {'train_time': None, 'valid_time': None}

    # restore model
    if config.restore_dir:
        shared_encoder_path = os.path.join(config.restore_dir, 'shared_encoder')
        decoder_G_path = os.path.join(config.restore_dir, 'decoder_G')
        decoder_D_path = os.path.join(config.restore_dir, 'decoder_D')
        model.restore(shared_encoder_path, decoder_G_path, decoder_D_path)
    # train model
    else:
        model_metrics = model.fit(x_train)

        # save model
        if config.save_dir:
            shared_encoder_path = os.path.join(config.save_dir, 'shared_encoder')
            decoder_G_path = os.path.join(config.save_dir, 'decoder_G')
            decoder_D_path = os.path.join(config.save_dir, 'decoder_D')
            model.save(shared_encoder_path, decoder_G_path, decoder_D_path)
            print(f'model saved in {config.save_dir}')
            print()
    
    # get train score
    predict_start = time.time()
    train_score = model.predict(x_train, alpha=config.alpha, beta=config.beta, on_dim=config.get_score_on_dim)
    model_metrics.update({
        'train_predict_time': time.time() - predict_start
    })
    if config.train_score_filename:
        np.save( os.path.join(config.result_dir, config.train_score_filename), train_score)
        paint(x_train[-len(train_score):].T, scores=[train_score], store_path=os.path.join(config.result_dir, 'train_score.png'))

    # get test score
    predict_start = time.time()
    test_score = model.predict(x_test, alpha=config.alpha, beta=config.beta, on_dim=config.get_score_on_dim)
    model_metrics.update({
        'test_predict_time': time.time() - predict_start
    })
    if config.test_score_filename:
        np.save(os.path.join(config.result_dir, config.test_score_filename), test_score)
        paint(x_test[-len(test_score):].T, scores=[test_score], store_path=os.path.join(config.result_dir, 'test_score.png'))
    

    pot = pot_detect(train_score, test_score, q=config.q, level=config.level)

    paint(x_test[-len(test_score):].T, scores=[test_score], thresholds=pot['thresholds'], store_path=os.path.join(config.result_dir, 'anomaly_detect.png'))
    model_metrics.update({
        'anomalies': pot['alarms']
    })
    
    with open(os.path.join(config.result_dir, 'model_result.json'), 'w') as f:
        json.dump(model_metrics, f)

        
if __name__ == '__main__':

    config = ConfigHandler().config
    print('Configuration:')
    print(config)
    print()
    main()
