from easydict import EasyDict as edict


__C                                             = edict()
# Consumers can get config by: from lstm_config import cfg

cfg                                             = __C

# DATASET options
__C.DATASET                                     = edict()

__C.DATASET.TRAIN_DATA_FOLDER_PATH              = "./../data/train"
__C.DATASET.VAL_DATA_FOLDER_PATH                = "./../data/valid"
__C.DATASET.TEST_DATA_FOLDER_PATH               = "./../data/test"
__C.DATASET.TRAIN_PROPORTION                    = 0.8
__C.DATASET.VALIDATION_PROPORTION               = 0.1
__C.DATASET.TRAINING_BATCH_SIZE                 = 8
__C.DATASET.VALIDATION_BATCH_SIZE               = 8
__C.DATASET.TEST_BATCH_SIZE                     = 8
__C.DATASET.NBR_CLASSE                          = 2

# TRAIN options
__C.TRAIN                                       = edict()

__C.TRAIN.LEARNING_RATE                         = 0.002
__C.TRAIN.NBR_EPOCH                             = 50
__C.TRAIN.CHECKPOINT_SAVE_PATH                  = './../models/try_7/'
__C.TRAIN.VALIDATION_RATIO                      = 2
__C.TRAIN.GRADIANT_ACCUMULATION                 = 2
__C.TRAIN.IMAGE_SHAPE                           = (3,1024,1024)
__C.TRAIN.PRETRAINED_WEIGHTS_PATH               = "./best_model.pth"

# EVALUATION options
__C.EVALUATION                                  = edict()

__C.EVALUATION.PRETRAINED_PATH                  = './../models/try_1/ckpt_19_metric_0.62096.ckpt'