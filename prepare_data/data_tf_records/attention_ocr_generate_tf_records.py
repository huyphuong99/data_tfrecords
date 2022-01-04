import glob
import os
import argparse
import shutil
import unicodedata

import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
import tqdm
from tensorflow.python.platform import gfile

NULL_CHAR = '<nul>'
LONGEST_TEXT = ''
CHARSET_CONTAINS = set()


def generate_charset(charset: str, output_path: str):
    charset = sorted(charset)
    output_file = output_path + "=" + str(len(charset) + 1) + ".txt"
    f = open(output_file, 'w')
    f.write(f"{0}\t{NULL_CHAR}\n")
    for i, c in enumerate(charset):
        f.write(f"{i + 1}\t{c}\n")
    f.close()
    return output_file


def get_char_mapping(charset_file):
    char_mapping = {}
    rev_char_mapping = {}
    with open(charset_file, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line[-1] != " ":
                id, char = line.split()
            else:
                id = line.strip()
                char = " "
            id = int(id)
            char_mapping[char] = id
            rev_char_mapping[id] = char
    if NULL_CHAR not in char_mapping:
        char_mapping[NULL_CHAR] = 200
        rev_char_mapping[200] = NULL_CHAR
    return char_mapping, rev_char_mapping


def encode_utf8_string(text, charset, length):
    for c in text:
        CHARSET_CONTAINS.add(c)
    char_ids_unpadded = []
    label = []
    for c in text:
        if c in charset:
            char_ids_unpadded.append(charset[c])
            label.append(c)
        else:
            continue
            raise Exception(f"Not found {c} in mapping")
    char_ids_padded = char_ids_unpadded + [charset[NULL_CHAR] for i in range(length - len(char_ids_unpadded))]
    label = ''.join(label)
    return label, char_ids_unpadded, char_ids_padded


def decode_string(ids, char_mapping):
    chars = [char_mapping[id] for id in ids]
    for id in ids:
        c = char_mapping[id]
        if c == NULL_CHAR:
            break
        chars.append(c)
    return ''.join(chars)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def letterbox_resize(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding

    :param
        image: origin image to be resize
        target_size: target image size,
            tuple of format (width, height).

    :returns
    """
    src_h, src_w, src_c = image.shape
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale_w = target_w / src_w
    scale_h = target_h / src_h

    if scale_w > scale_h:
        new_h = target_h
        new_w = int(scale_h * src_w)
    else:
        return cv2.resize(image, (target_w, target_h))

    mask = np.zeros(shape=(target_h, target_w, src_c), dtype=np.uint8)
    tmp_img = cv2.resize(image, (new_w, new_h))
    tmp_img = np.reshape(tmp_img, (new_h, new_w, src_c))
    mask[0:new_h, 0:new_w] = tmp_img
    return mask


def _tf_example(img_file, label, char_mapping, image_shape, max_str_len, num_of_views):
    w, h, c = image_shape
    img_array = cv2.imread(img_file)
    if img_array is None:
        print("Error:", img_file)
        return
    if c == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = letterbox_resize(img_array, (w, h))
    if DEBUG:
        cv2.imshow("test", img_array)
        cv2.waitKey(0)
    retval, image = cv2.imencode('.png', img_array)
    image_data = image.tostring()
    # image = tf.image.convert_image_dtype(img_array, dtype=tf.uint8)
    # image = tf.image.encode_png(image)
    # image_data = sess.run(image)
    # img = gfile.FastGFile(img_file, 'rb').read()

    label, char_ids_unpadded, char_ids_padded = encode_utf8_string(label, char_mapping, max_str_len)

    features = tf.train.Features(feature={
        'image/format': _bytes_feature([b"PNG"]),
        'image/encoded': _bytes_feature([image_data]),
        'image/class': _int64_feature(char_ids_padded),
        'image/unpadded_class': _int64_feature(char_ids_unpadded),
        'image/width': _int64_feature([img_array.shape[1]]),
        'image/orig_width': _int64_feature([int(img_array.shape[1] / num_of_views)]),
        'image/text': _bytes_feature([label.encode('utf-8')])
    })

    example = tf.train.Example(features=features)
    return example


def get_labels_in_filename(img_dirs: list):
    file_paths = []
    labels = []
    for dir in img_dirs:
        for file in glob.glob(dir + "/*.png"):
            filename = os.path.basename(file)[:-4]
            img_index, info_index, label = filename.split("_")
            file_paths.append(file)
            labels.append(label)
    return pd.DataFrame({"filename": file_paths, "label": labels})


def gen_code_data(output_dir, data_name, traing_size, test_size, charset_filename, image_shape, num_of_views,
                  max_sequence_length, null_code):
    code = f"""from . import fsns

DEFAULT_DATASET_DIR = 'datasets/data/{data_name}'

DEFAULT_CONFIG = {{
    'name':
        '{data_name}',
    'splits': {{
        'train': {{
            'size': {traing_size},
            'pattern': 'train*'
        }},
        'test': {{
            'size': {test_size},
            'pattern': 'test*'
        }}
    }},
    'charset_filename':
        '{charset_filename}',
    'image_shape': ({image_shape[1]}, {image_shape[0]}, {image_shape[2]}),
    'num_of_views':
        {num_of_views},
    'max_sequence_length':
        {max_sequence_length},
    'null_code':
        {null_code},
    'items_to_descriptions': {{
        'image':
            'A [150 x 600 x 3] color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }}
}}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config) 
"""
    print(code, file=open(os.path.join(output_dir, "{}.py".format(data_name)), 'w'))


def tf_records(img_dir, labels_file, output_dir, data_name, charset_file, image_shape, max_len, test_size=0.2):
    char_mapping, rev_char_mapping = get_char_mapping(charset_file)

    output_dir_data = os.path.join(output_dir, 'data', data_name)
    os.makedirs(output_dir_data, exist_ok=True)
    train_file = os.path.join(output_dir_data, 'train.tfrecord')
    test_file = os.path.join(output_dir_data, 'test.tfrecord')
    charset_filename = os.path.basename(charset_file)
    output_charset_file = os.path.join(output_dir_data, charset_filename)
    if not os.path.exists(output_charset_file):
        shutil.copyfile(charset_file, output_charset_file)

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    train_writer = tf.io.TFRecordWriter(train_file)
    test_writer = tf.io.TFRecordWriter(test_file)

    if labels_file is not None:
        label_df = pd.read_csv(labels_file, header=0, names=['filename', 'label'],
                               dtype={"label": str, "filename": str})

    else:
        label_df = get_labels_in_filename(img_dir)

    label_df = label_df.sample(frac=1)
    n_train = int(len(label_df) * (1 - test_size))
    n_test = len(label_df) - n_train

    print("Split data: {} for trains and {} for test".format(n_train, n_test))

    global LONGEST_TEXT

    for i, row in tqdm.tqdm(label_df.iterrows()):
        img_path = row['filename']
        if not img_path.startswith('/'):
            for dir in img_dir:
                img_path = os.path.join(dir, os.path.basename(row['filename']))
                if os.path.exists(img_path):
                    break
        else:
            for dir in img_dir:
                img_path = os.path.join(dir, os.path.basename(row['filename']))
                if os.path.exists(img_path):
                    break
        label = row['label']
        try:
            label = unicodedata.normalize('NFKC', label)
        except:
            print(row)
            exit(0)
        # label = str(label).lower().replace("#", "/")
        if len(LONGEST_TEXT) < len(label):
            LONGEST_TEXT = label
            print(list(LONGEST_TEXT))

        example = _tf_example(img_file=img_path,
                              label=label,
                              char_mapping=char_mapping,
                              image_shape=image_shape,
                              max_str_len=max_len,
                              num_of_views=1)
        if example is None:
            continue

        if i < n_train:
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

    train_writer.close()
    test_writer.close()
    gen_code_data(output_dir=output_dir,
                  data_name=data_name,
                  traing_size=n_train,
                  test_size=n_test,
                  charset_filename=charset_filename,
                  image_shape=image_shape,
                  num_of_views=1,
                  max_sequence_length=max_len,
                  null_code=char_mapping[NULL_CHAR]
                  )

    print("============> Complete export tfrecords file <==============")
    print("Longest text: '{}' has length is {}".format(LONGEST_TEXT, len(LONGEST_TEXT)))
    print("CHARSET CONTAINS: {} characters: {}".format(len(CHARSET_CONTAINS), sorted(CHARSET_CONTAINS)))
    print("Train file has: {:6} samples, that is saved at {}".format(n_train, train_file))
    print("Test file has : {:6} samples, that is saved at {}".format(n_test, test_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', required=True)
    parser.add_argument('-l', '--label', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-d', '--dataset_name', required=True)
    parser.add_argument('-c', '--charset', required=True)
    parser.add_argument('-s', '--image_shape', required=True, type=tuple, default=(625, 75, 3),
                        help="default (625, 75, 3)")
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)

    args = parser.parse_args()

    tf_records(args.img_dir,
               args.label,
               args.output_dir,
               args.data_name,
               args.charset,
               args.image_shape,
               args.max_len,
               args.test_size)


if __name__ == '__main__':
    # main()
    # charset_file = generate_charset(
    #     " .-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    #     "/home/huyphuong99/PycharmProjects/data_tfrecords/charsets/vr_plate")
    # print(charset_file)

    # exit(0)
    DEBUG = False
    # tf_records(["/media/data_it/Data_set/database_image/card/vr/info/date",
    #             "/media/data_it/Data_set/database_image/card/vr/info/month",
    #             "/media/data_it/Data_set/database_image/card/vr/info/year",
    #             "/media/data_it/Data_set/database_image/card/vr/info/capacity",
    #             "/media/data_it/Data_set/database_image/card/vr/info/seat_capacity",
    #             "/media/data_it/Data_set/database_image/card/vr/info/registration_date",
    #             ],
    #            labels_file="/media/data_it/Data_set/database_image/card/vr/info/date_number.csv",
    #            output_dir="/media/data_it/thiennt/cv_end_to_end/training/tensorflow/models/research/attention_ocr/python/datasets",
    #            data_name="VR_DATE_NUMBER",
    #            charset_file="/media/data_it/thiennt/cv_end_to_end/training/ocr/data_prepare/charsets/date=14.txt",
    #            image_shape=(225, 75, 3),
    #            max_len=10,
    #            test_size=0.05)
    path = "/media/huyphuong99/huyphuong99/tima/project/vr/info_vr/REFORMAT_DATA/PLATE/"
    data_name = 'plate'
    tf_records([f"{path}{data_name}",],
               labels_file=f"{path}{data_name}.csv",
               output_dir="/home/huyphuong99/PycharmProjects/data_tfrecords/datasets/data_attention",
               data_name=f"vr_{data_name}",
               charset_file="/home/huyphuong99/PycharmProjects/data_tfrecords/charsets/vr_plate=40.txt",
               image_shape=(400, 75, 3),
               max_len=12,
               test_size=0.1)
