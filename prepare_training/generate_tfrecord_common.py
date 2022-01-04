import logging
import os
import traceback
import contextlib2
import tensorflow.compat.v1 as tf
import tqdm
from object_detection.utils import dataset_util, label_map_util
from base_object import ObjectsImage, LabelSource, get_category_mapping


def clip_to_unit(x):
    return min(max(x, 0.0), 1.0)


def create_tf_example(annotations: ObjectsImage,
                      include_keypoint: bool = False,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.

    Args:
      annotations: ObjectsImage
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
      include_keypoint:
    Returns:
      example: The converted tf.Example

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    feature_dict = {
        'image/height': dataset_util.int64_feature(annotations.img_height),
        'image/width': dataset_util.int64_feature(annotations.img_width),
        'image/filename': dataset_util.bytes_feature(annotations.filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(str(annotations.source_id).encode('utf8')),
        # 'image/key/sha256': dataset_util.bytes_feature(annotations.sha_key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(annotations.encoded_jpg),
        'image/format': dataset_util.bytes_feature(annotations.format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(annotations.xmin_norm),
        'image/object/bbox/xmax': dataset_util.float_list_feature(annotations.xmax_norm),
        'image/object/bbox/ymin': dataset_util.float_list_feature(annotations.ymin_norm),
        'image/object/bbox/ymax': dataset_util.float_list_feature(annotations.ymax_norm),
        'image/object/class/text': dataset_util.bytes_list_feature(annotations.labels),
        'image/object/class/label': dataset_util.int64_list_feature(annotations.classes)
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(annotations.encoded_mask_png))  # TODO: add mask
    if include_keypoint:
        feature_dict['image/object/keypoint/x'] = (
            dataset_util.float_list_feature(annotations.keypoints_x))
        feature_dict['image/object/keypoint/y'] = (
            dataset_util.float_list_feature(annotations.keypoints_y))
        feature_dict['image/object/keypoint/num'] = (
            dataset_util.int64_list_feature(annotations.num_keypoints))
        feature_dict['image/object/keypoint/visibility'] = (
            dataset_util.int64_list_feature(annotations.keypoints_visibility))
        feature_dict['image/object/keypoint/text'] = (
            dataset_util.bytes_list_feature(annotations.keypoints_name))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards

    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def _create_tf_record(annotation_file: str,
                      label_names_file: str,
                      label_source: LabelSource,
                      image_dir: str,
                      output_path: str,
                      include_masks: bool = False,
                      include_keypoint: bool = False,
                      num_shards: int = 1):
    """ Load labelme annotation json files and converts to tf.Record format.

    :param annotations_file: text file, each line is name of file label
    :param image_dir: Directory containing the image files.
    :param output_path: Path to output tf.Record fil .
    :param num_shards: number of output file shards.
    :return:
    """
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(annotation_file, 'r') as fid:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        cat2idx, idx2cat = get_category_mapping(label_names_file)

        for i, line in tqdm.tqdm(enumerate(fid)):
            if line.startswith("#"):
                continue
            label_path = line.strip()
            if label_source == LabelSource.LABEL_ME:
                annotation = ObjectsImage.get_from_labelme(label_path, image_dir, cat2idx, idx2cat)
            elif label_source == LabelSource.LABEL_VOC:
                annotation = ObjectsImage.get_from_voc(label_path, image_dir, cat2idx, idx2cat)
            elif label_source == LabelSource.LABEL_YOLO:
                annotation = ObjectsImage.get_from_yolo(label_path, image_dir, cat2idx, idx2cat)
            else:
                raise Exception(f"Can't process source {label_source}")
            try:
                tf_example = create_tf_example(annotation, include_keypoint=include_keypoint,
                                               include_masks=include_masks)
                shard_idx = i % num_shards
                if tf_example:
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error at:", label_path)
                traceback.print_exc()


if __name__ == '__main__':
    #CHÚ Ý NẾU SỬ DỤNG TÊN LABEL KHÁC THÌ PHẢI ĐỔI TÊN LABEL TRONG FILE BASE ĐỂ KHỚP VỚI TÊN MÌNH ĐẶT
    train_or_test = "train"
    file = f"{train_or_test}_cropped_id_221021"
    file_ano = f"{file}.txt"
    label_source = LabelSource.LABEL_VOC
    image_dir = f"/home/huyphuong99/Desktop/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_221021/partion_data/{file}"
    workdir = f"/home/huyphuong99/Desktop/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_221021/file_necessary"
    include_keypoint = True if label_source == LabelSource.LABEL_ME else False
    num_shards = 1
    annotation_file = os.path.join(workdir, f"file_annotation/{file_ano}")
    label_names_file = os.path.join(workdir, "label_map.pbtxt")
    output_path = os.path.join(workdir, f'file_record/{file}.record')

    print("===========================================================================================================")
    print(f"Project:")
    print(f"Data:\t\t")
    print(f"Data_type:\t\t")
    print(f"Label source:\t\t{label_source}")
    print(f"Keypoint:\t\t{include_keypoint}")
    print(f"Image Dir:\t\t{image_dir}")
    print(f"Output:\t\t{num_shards} shards. At: {output_path}")
    print("===========================================================================================================")
    try:
        _create_tf_record(
            annotation_file=annotation_file,
            label_names_file=label_names_file,
            label_source=label_source,
            image_dir=image_dir,
            output_path=output_path,
            include_keypoint=include_keypoint,
            include_masks=False,
            num_shards=num_shards)
    except Exception as e:
        raise e
