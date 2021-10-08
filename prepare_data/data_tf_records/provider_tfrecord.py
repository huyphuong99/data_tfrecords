import os
import tensorflow as tf
from debugpy.common.compat import filename

from config import DataRecognitionConfig as cf
from config import TrainingConfig as tcf
import matplotlib.pyplot as plt
import numpy as np



def get_data(data: str = "train") -> tf.data.Dataset:
    """
    Args:
      data: train/test

    Returns:
      datasets : tf.data.Dataset object.
                elements structured as [features, labels]
                Example feature structure can be seen in postbatch_fn
    """
    # Get raw data
    if data == "train":
        file_patterns = cf.train_file_patterns
    else:
        file_patterns = cf.test_file_patterns
    dataset = get_dataset(cf.data_dir, file_patterns)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(postbatch_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # for data in dataset.as_numpy_iterator():
    #     print(data)
    # dataset = dataset.map(add_padd, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # dataset = dataset.batch(tcf.batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print(dataset)
    #
    dataset = dataset.padded_batch(tcf.batch_size, padded_shapes={
        "image": [cf.height, None, cf.depth],
        "label": [None],
        "filename": []
    }, padding_values={
        "image": 0.0,
        "label": len(cf.charset) + 1,
        "filename": None
    })
    return dataset


def get_dataset(base_dir, file_patterns):
    """ Get a Dataset from TFRecord files.
    Parameters:
      base_dir      : Directory containing the TFRecord files
      file_patterns : List of wildcard patterns for TFRecord files to read
    Returns:
      image   : preprocessed image
                  tf.float32 tensor of shape [32, ?, 1] (? = width)
      width   : width (in pixels) of image
                  tf.int32 tensor of shape []
      labels  : list of indices of characters.txt.txt mapping text->out_charset
                  tf.int32 tensor of shape [?] (? = length+1)
      length  : length of labels (sans -1 EOS token)
                  tf.int32 tensor of shape []
      text    : ground truth string
                  tf.string tensor of shape []
    """
    # Get filenames as list of tensors
    tensor_filenames = _get_filenames(base_dir, file_patterns)
    # Get filenames into a datasets format
    ds_filenames = tf.data.Dataset.from_tensor_slices(tensor_filenames)
    dataset = tf.data.TFRecordDataset(ds_filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    return dataset


def _get_filenames(base_dir, file_patterns=['*.tfrecord']):
    """Get a list of record files"""
    # List of lists ...
    data_files = [tf.io.gfile.glob(os.path.join(base_dir, file_pattern)) for file_pattern in file_patterns]
    data_files = [data_file for sublist in data_files for data_file in sublist]
    print(f"Load data from: {data_files}")
    return data_files


rng = tf.random.Generator.from_seed(123, alg="philox")


def preprocess_fn(data):
    print(type(data))
    feature_map = {
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/labels": tf.io.VarLenFeature(dtype=tf.int64),
        # "image/height": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1),
        "image/width": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1),
        "image/filename": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "text/string": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "text/length": tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1)
    }
    features = tf.io.parse_single_example(data, feature_map)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3 if cf.depth == 3 else 1)
    # height = tf.cast(features["image/height", tf.int32])
    # print(height)
    width = tf.cast(features["image/width"], tf.int32)
    label = tf.cast(features["image/labels"], tf.int32)
    length = features["text/length"]
    text = features["text/string"]
    image = preprocess_image(image)
    seed = rng.make_seeds(2)[0]
    if tcf.augment_data:
        image = augment_image(image, seed)
    name_img = tf.cast(features["image/filename"], tf.string)
    return image, width, label, length, text, name_img


def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)
    image = tf.image.resize(image, [cf.height, cf.max_width], preserve_aspect_ratio=True)
    return image


def augment_image(image, seed):
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    image = tf.image.stateless_random_contrast(image, lower=0.4, upper=0.6, seed=new_seed)
    return image


def postbatch_fn(image, width, label, length, text, filename):
    label = tf.sparse.to_dense(label)
    features = {
        "image": image,
        "width": width,
        "length": length,
        "text": text,
        "filemame": filename
    }
    return {"image": image, "label": label, "filename": filename}


if __name__ == "__main__":
    base_dir = ""
    type_data = "passport_passport_train"
    file_patterns = [f"*{type_data}.records*"]
    dataset = get_data("train")

    for data in dataset.as_numpy_iterator():
        print(data["filename"].shape)
        break
    #     print(data["image"])
    #     # print(data["image"].shape)
    #     print(data['label'])
    #     break
    # for i in range(len(data["image"])):
    #     image = (np.array(data['image'][i]) + 0.5) * 255.0
    #     image = image.astype(np.uint8)
    #     plt.imshow(image)
    #     label = "".join(data['label'][i].astype(str)).replace(f"{len(cf.charset)+1}", "")
    #     plt.title(label)
    #     plt.show()
    #     break
