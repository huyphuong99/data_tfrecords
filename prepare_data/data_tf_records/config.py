import json
import os


class DataRecognitionConfig:
    data_name: str = ""
    img_dir: str = ""
    label_file: str = ""
    work_dir: str = "/home/huyphuong99/PycharmProjects/passport_tima/datasets/file_tfrecord"
    data_dir: str = "/home/huyphuong99/PycharmProjects/passport_tima/datasets/file_tfrecord/annotations"
    train_file_patterns: list = ["*passport_train.records*"]
    test_file_patterns: list = ["*passport_test.records*"]
    height, depth, max_width, max_len = 50, 3, 250, 12
    charset = sorted("BCN0123456789")
    test_size = 0


class TrainingConfig:
    batch_size = 64
    augment_data = False

class CnnConfig:
    type_model = 'stack'
    layers = [
        {'filter': 64, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid',
         'batch_norm': True},
        {'filter': 128, 'kernel_size': 3, 'padding': 'same', 'strides': 1, 'pool_size': 2, 'padding_pool': 'valid',
         'batch_norm': True},
    ]


class RnnConfig:
    input_depth = 64
    input_dropout = 0.2
    layers = [
        {'units': 128, 'dropout': 0.25},
        {'units': 64, 'dropout': 0.25}
    ]


class HeadConfig:
    classes = len(DataRecognitionConfig.charset) + 1



def load_config(config_path: str):
    config = json.load(open(config_path, 'r'))
    data_config = config['DataRecognitionConfig']
    DataRecognitionConfig.data_name = data_config['data_name']
    DataRecognitionConfig.img_dir = data_config['image_dir']
    DataRecognitionConfig.label_file = data_config['label_file']
    DataRecognitionConfig.work_dir = data_config['work_dir']
    DataRecognitionConfig.data_dir = os.path.join(DataRecognitionConfig.work_dir,'')
    DataRecognitionConfig.train_file_patterns = data_config['train_file_patterns']
    DataRecognitionConfig.test_file_patterns = data_config['test_file_patterns']
    DataRecognitionConfig.height = data_config['height']
    DataRecognitionConfig.depth = data_config['depth']
    DataRecognitionConfig.max_width = data_config['max_width']
    DataRecognitionConfig.charset = sorted(data_config['charset'])
    DataRecognitionConfig.max_len = data_config['max_len']
    DataRecognitionConfig.test_size = data_config['test_size']

    cnn_config = config['CnnConfig']
    CnnConfig.type_model = cnn_config['type_model']
    CnnConfig.layers = cnn_config.get('layers')

    rnn_config = config['RnnConfig']
    RnnConfig.input_depth = rnn_config['input_depth']
    RnnConfig.input_dropout = rnn_config['input_dropout']
    RnnConfig.layers = rnn_config['layers']

    HeadConfig.classes = len(DataRecognitionConfig.charset) + 1

    training_config = config['TrainingConfig']
    TrainingConfig.augment_data = training_config['augment_data']
    TrainingConfig.optimizer = training_config['optimizer']
    TrainingConfig.learning_rate = training_config['learning_rate']
    TrainingConfig.epochs = training_config['epochs']
    TrainingConfig.checkpoints = os.path.join(DataRecognitionConfig.work_dir, 'models')
    TrainingConfig.batch_size = training_config['batch_size']
