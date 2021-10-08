import io
import json
import math
from enum import Enum
import os
import xml.etree.ElementTree as ET
from typing import List, Dict
import numpy as np
from google.protobuf import text_format
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, \
    StringIntLabelMapItem
import PIL
import tensorflow.compat.v1 as tf
import cv2


class LabelSource(Enum):
    LABEL_ME = 1
    LABEL_VOC = 2
    LABEL_YOLO = 3


def points_to_bbox(points):
    x = [t[0] for t in points]
    y = [t[1] for t in points]
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    return x_min, y_min, x_max, y_max


def get_category_mapping(label_names_file):
    """Creates dictionary of COCO compatible categories keyed by category file_tfrecord.

      Returns:
        category_index: a dict containing the same entries as categories, but keyed
          by the 'file_tfrecord' field of each category.
      """
    cat2idx = {}
    idx2cat = {}
    with open(label_names_file, 'r') as f:
        label_maps = text_format.Parse(f.read(), StringIntLabelMap())
        for label_map in label_maps.item:
            cat2idx[label_map.name] = label_map.id
            idx2cat[label_map.id] = label_map.name
    return cat2idx, idx2cat


class BaseObject:
    def __init__(self, label: str, label_id: int, xmin: int, xmax: int, ymin: int, ymax: int,
                 points: List = None):
        self.label = label
        self.label_idx = label_id
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.points = points
        if (self.xmin is None or self.xmax is None or self.ymin is None or self.ymax is None) \
                and self.points is not None and len(self.points) > 3:
            self.xmin, self.ymin, self.xmax, self.ymax = points_to_bbox(self.points)
        self.h = self.ymax - self.ymin
        self.w = self.xmax - self.xmin

    @staticmethod
    def check(val_check):
        val_check = val_check
        if val_check < 0:
            val_check = 0
        elif val_check > 1:
            val_check = 1
        return val_check


class ObjectsImage:
    def __init__(self, filename: str, file_path: str, img_width: int, img_height: int,
                 img_channel: int,
                 base64_img: str, source_id: str = None,
                 objects: List[BaseObject] = None, encoded_jpg=None):
        self.source_id = source_id
        if self.source_id is None:
            self.source_id = filename
        self.filename = filename
        self.file_path = file_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.base64_img = base64_img
        self.format = b'jpg'
        self.objects = objects
        self.encoded_jpg = encoded_jpg
        self.check = BaseObject.check
        if self.encoded_jpg is None:
            with tf.gfile.GFile(file_path, 'rb') as fid:
                self.encoded_jpg = fid.read()
        # self.sha_key = hashlib.sha256(self.encoded_jpg).hexdigest()

        self.xmin_norm = []
        self.ymin_norm = []
        self.xmax_norm = []
        self.ymax_norm = []
        self.labels = []
        self.classes = []

        _KEYPOINT_NAMES = [b'top_left', b'top_right', b'bottom_right', b'bottom_left']
        self.keypoints_x = []
        self.keypoints_y = []
        self.keypoints_visibility = []
        self.keypoints_name = []
        self.num_keypoints = []

        for obj in objects:
            if obj.xmin / self.img_width > 1 or obj.xmin / self.img_width < 0 or obj.xmax / self.img_width < 0 or obj.xmax / self.img_width > 1 or obj.ymin / self.img_height < 0 or obj.ymin / self.img_height > 1 or obj.ymax / self.img_height < 0 or obj.ymax / self.img_height > 1:
                print(self.filename)
            self.xmin_norm.append(self.check(float(obj.xmin) / self.img_width))
            self.ymin_norm.append(self.check(float(obj.ymin) / self.img_height))
            self.xmax_norm.append(self.check(float(obj.xmax) / self.img_width))
            self.ymax_norm.append(self.check(float(obj.ymax) / self.img_height))
            self.labels.append(obj.label.encode('utf8'))
            self.classes.append(obj.label_idx)

            if obj.points is not None:
                for point in obj.points:
                    self.keypoints_x.append(self.check(float(point[0] / self.img_width)))
                    self.keypoints_y.append(self.check(float(point[1] / self.img_height)))
                    self.keypoints_visibility.append(1)  # 1 for visibility and 0 for otherwise

                n_has_k = len(self.keypoints_x)
                if n_has_k < 4:
                    for i in range(0, 4 - n_has_k):
                        self.keypoints_x.append(0.0)
                        self.keypoints_y.append(0.0)
                        self.keypoints_visibility.append(0)  # 1 for visibility and 0 for otherwise

                self.keypoints_name.extend(_KEYPOINT_NAMES)  # FIXME: rename
                self.num_keypoints.append(len(obj.points))

    @staticmethod
    def get_from_labelme(label_path: str, img_dir: str, cat2idx: Dict, idx2cat: Dict):
        with tf.gfile.GFile(label_path, 'r') as f:
            annotations = json.load(f)
            img_height = annotations['imageHeight']
            img_width = annotations['imageWidth']
            filename = os.path.basename(annotations['imagePath'])
            base64_image = annotations['imageData']
            file_path = os.path.join(img_dir, filename)

            objects = []
            for obj in annotations['shapes']:
                xmin, ymin, xmax, ymax = None, None, None, None
                points = None
                label = obj['label']
                label = 'card'
                if label not in cat2idx:
                    continue
                label_id = cat2idx[label]
                if obj['shape_type'] == 'rectangle':
                    xmin = obj['points'][0][0]
                    ymin = obj['points'][0][1]
                    xmax = obj['points'][1][0]
                    ymax = obj['points'][1][1]
                elif obj['shape_type'] == 'polygon':
                    points = obj['points']
                # if xmax is None or xmin is None or ymax is None or ymin is None:
                #     print(filename)
                base_obj = BaseObject(label=label,
                                      label_id=label_id,
                                      xmin=xmin,
                                      ymin=ymin,
                                      xmax=xmax,
                                      ymax=ymax,
                                      points=points)
                objects.append(base_obj)

            return ObjectsImage(filename=filename, file_path=file_path, img_width=img_width,
                                img_height=img_height,
                                img_channel=3, base64_img=base64_image, objects=objects)

    @staticmethod
    def get_from_voc(label_path: str, img_dir: str, cat2idx: Dict, idx2cat: Dict):
        tree = ET.parse(label_path)
        root = tree.getroot()
        filename = root.find('filename').text
        file_path = os.path.join(img_dir, filename)
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        img_channel = int(root.find('size/depth').text)

        objects = []
        for obj in root.iter('object'):
            ymin = int(obj.find("bndbox/ymin").text)
            xmin = int(obj.find("bndbox/xmin").text)
            ymax = int(obj.find("bndbox/ymax").text)
            xmax = int(obj.find("bndbox/xmax").text)
            label = obj.find("name").text
            if label not in cat2idx:
                continue
            label_id = cat2idx[label]
            objects.append(BaseObject(label=label,
                                      label_id=label_id,
                                      xmin=xmin,
                                      ymin=ymin,
                                      xmax=xmax,
                                      ymax=ymax))

        return ObjectsImage(filename=filename,
                            file_path=file_path,
                            img_width=img_width,
                            img_height=img_height,
                            img_channel=img_channel,
                            base64_img=None,
                            objects=objects)

    @staticmethod
    def get_from_yolo(label_path: str, img_dir: str, cat2idx: Dict, idx2cat: Dict):
        filename = os.path.basename(label_path)
        file_path = os.path.join(img_dir,
                                 filename)  # FIXME: filename and file path of image( not label)
        with tf.gfile.GFile(file_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        img_width, img_height = image.size

        objects = []
        for line in open(label_path, 'r'):
            line = line.strip()
            if len(line) < 4:
                continue
            numbers = line.split(" ")
            label_id = int(numbers[0])
            if label_id not in idx2cat:
                continue
            label = idx2cat[label_id].encode('utf8')
            xc, yc, w, h = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(
                numbers[4])
            xmin = ((2 * xc * img_width) - (w * img_width)) / 2
            xmax = ((2 * xc * img_width) + (w * img_width)) / 2
            ymin = ((2 * yc * img_height) - (h * img_height)) / 2
            ymax = ((2 * yc * img_height) + (h * img_height)) / 2

            try:
                objects.append(BaseObject(label=label,
                                          label_id=label_id,
                                          xmin=xmin,
                                          ymin=ymin,
                                          xmax=xmax,
                                          ymax=ymax))
            except:
                print(f"File do not exist")

        return ObjectsImage(filename=filename,
                            file_path=file_path,
                            img_width=img_width,
                            img_height=img_height,
                            img_channel=3,
                            base64_img=None,
                            objects=objects,
                            encoded_jpg=encoded_jpg)


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def crop_with_point(image, p1, p2, p3, p4):
    width = int(max(distance(p1, p2), distance(p3, p4)))
    height = int(max(distance(p1, p4), distance(p3, p2)))
    # height = int(width * 1.4143)  # TODO:

    # x1 = min(p1[0], p2[0], p3[0], p4[0])
    # x2 = max(p1[0], p2[0], p3[0], p4[0])
    # y1 = min(p1[1], p2[1], p3[1], p4[1])
    # y2 = max(p1[1], p2[1], p3[1], p4[1])
    # width = x2 - x1
    # height = y2 - y1
    # tmp1 = max(width, height)
    # tmp2 = min(width, height)
    # width = tmp1
    # height = tmp2

    src = [
        p1, p2, p3, p4
    ]

    src = np.array(src, dtype=np.float32)

    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    wrarped = cv2.warpPerspective(image, m, (width, height))
    return wrarped


def crop_with_rect(image: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int):
    return image[ymin:ymax, xmin: xmax]


def crop_with_annotation(img: np.ndarray, annotation: BaseObject):
    if annotation.points is not None:
        return crop_with_point(img, *annotation.points)
    else:
        return crop_with_rect(img, annotation.xmin, annotation.ymin, annotation.xmax,
                              annotation.ymax)
