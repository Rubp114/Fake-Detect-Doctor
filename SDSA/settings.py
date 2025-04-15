import logging
import os.path as osp
import time
import numpy as np

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

DATASET = 'Weibo'

if DATASET == 'Weibo':
    NUM_EPOCH = 300
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 1
    K = 1
    ETA = 0.1  # 0.1
    ALPHA = 0.4

if DATASET == 'Fakeddit':
    NUM_EPOCH = 100
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 1
    K = 1
    ETA = 0.1  # 0.1
    ALPHA = 0.5

BATCH_SIZE = 32
CODE_LEN = 512
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4
event = 10
GPU_ID = 0
NUM_WORKERS = 0
EPOCH_INTERVAL = 2
Best_num = 0.85
MODEL_DIR = './checkpoint'
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('K = %.4f' % K)
logger.info('ETA = %.4f' % ETA)


logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)


logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)

logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)

logger.info('--------------------------------------------------------------------')
