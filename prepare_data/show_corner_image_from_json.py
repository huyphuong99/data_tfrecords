import glob
import os
import json
import numpy as np
import cv2
from crop_image_four_point import four_point_transform
path_file_json = "/media/huyphuong/huyphuong99/tima/project/engine/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/new_cccd_150621/back_150621"
path_ouput = "//media/huyphuong/huyphuong99/tima/project/engine/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_150621/cropped_back"

ext = ['png', 'jpg', 'gif', "PNG", "jpeg"]
files = []
[files.extend(glob.glob(os.path.join(path_file_json,f"*.{e}"))) for e in ext]
name_json = [list(set([files[i].replace(ext[j], "json")  for j in range(len(ext))])) for i in range(len(files))]
for i in range(len(name_json)):
    img, f, name = None, None, None
    try:
        name = os.path.basename(name_json[i][0])
        img = cv2.imread(name_json[i][0])
        f = open(name_json[i][1])
        f = json.load(f)
    except:
        name = os.path.basename(name_json[i][1])
        img = cv2.imread(name_json[i][1])
        f = open(name_json[i][0])
        f = json.load(f)
    coordinate_box = f["shapes"][0]["points"]
    pts = np.array(coordinate_box, dtype='float32')
    img = four_point_transform(img, pts)
    h, w, c = img.shape
    cv2.imwrite(f"{path_ouput}/cr_{name}", img)
    # cv2.imshow("name", img)
    # cv2.waitKey()

# images = [cv2.imread(file) for file in files]
# for file in sorted(glob.glob(os.path.join(path_file_json, "*.json"))):
#     filename = os.path.basename(file)
#     f = open(file)
#     f = json.load(f)
#
#     f["shapes"][0]["label"] = "passport"
#     with open(file, "w") as _file:
#         json.dump(f, _file)
#
#     if f["shapes"][0]["shape_type"] == "rectangle":
#         os.remove(file)
#         print(filename)
#     elif f["shapes"][0]["shape_type"] == "polygon":
#         path_img = os.path.join(path_file_json, filename.replace(".json", ".jpg"))
#         name = os.path.basename(path_img)
#         coordinate_box = f["shapes"][0]["points"]
#         coordinate_box = np.array(coordinate_box, dtype='float32')
#         img = cv2.imread(path_img)
#         try:
#             img = four_point_transform(img, coordinate_box)
#             # cv2.imshow(f"name", img)
#             # cv2.waitKey()
#         except:
#             os.remove(file)
#             print(f"file error{name}")
