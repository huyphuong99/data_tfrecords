import time

import tensorflow.compat.v1 as tf
import glob
import os

tfrecord_path = "/home/huyphuong99/PycharmProjects/passport_tima/datasets/file_tfrecord/vr/plate/"
total_images = 0
train_files = sorted(glob.glob(os.path.join(tfrecord_path, '*vr*')))
print(train_files)
for idx, file in enumerate(train_files):
    try:
        total_images += sum([1 for _ in tf.io.tf_record_iterator(file)]) # Check corrupted tf records
    except:
        print("{}: {} is corrupted".format(idx, file))
print("Succeed, no corrupted tf records found for {} images".format(total_images))
i = 0
for record in tf.io.tf_record_iterator(train_files[0]):
    i += 1
    tf_example = tf.train.Example.FromString(record)
    print(type(tf_example), tf_example)
    print(i)
    break
    # if i == 10:
    #     break
