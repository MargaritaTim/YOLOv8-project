# ** Imports **
import image_properties_functions as image_utils
import functions as utils

import os
from os import listdir

import random
import shutil
import struct
import re
import locale

import numpy as np
import pandas as pd

from matplotlib import pyplot
import cv2
# from google.colab.patches import cv2_imshow

import xml.etree.ElementTree as ET

import torch
import torchvision.models as models

import ultralytics
from ultralytics import YOLO

torch.manual_seed(0)

# **YOLOv8 model**

# **1 Create YOLOv8 model**
locale.getpreferredencoding = lambda: "UTF-8"

# !pip install pyyaml h5py

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# model.train(data='coco128.yaml', epochs=10, imgsz=640)

# **2 Save the trained model**
# export the model
# model.export()  

# **3 Load trained model & example**
model_trained = YOLO(utils.repo_image_path('/best.torchscript'), task='detect')

# ** Example**
image_path = utils.repo_image_path('/Kangaroos/00050.jpg')
utils.predict_plot_image(image_path,model_trained)

# **Predict COCO128**
coco128_path = utils.repo_image_path('/coco128/image')
coco_annos_dir = utils.repo_image_path('/coco128/annotations')

df_coco, coco_iou = utils.pipeline('coco128', coco128_path, coco_annos_dir, 'jpg', model_trained)

print(df_coco)
print(coco_iou)

# **Predict Mouse Dataset**

mouse_path = utils.repo_image_path('/Mouse')

mouse_annos_dir = utils.repo_image_path('/Mouse/annotations')

df_mouse, mouse_iou = utils.pipeline('mouse', mouse_path, mouse_annos_dir, 'jpg', model_trained)

print(mouse_iou)

"""Print images with low score"""

df_mouse_low_score = df_mouse[(df_mouse["avg_score"] < 0.5)].sort_values(by=['avg_score'])

#df_mouse_low_score

image_list = df_mouse_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Mouse", model_trained)

"""Print images with high score"""

df_mouse_high_score = df_mouse[(df_mouse["avg_score"] > 0.8)].sort_values(by=['avg_score'])

image_list = df_mouse_high_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Mouse",model_trained)

"""# **Predict Zebras Dataset**"""

zebra_image_path = utils.repo_image_path('/Zebra')

zebra_annos_dir = utils.repo_image_path('/Zebra/annotations')

df_zebra, zebra_iou = utils.pipeline('zebra', zebra_image_path, zebra_annos_dir, 'jpg', model_trained)

print(zebra_iou)

"""Print low score images"""

df_zebra_low_score = df_zebra[(df_zebra["avg_score"] < 0.5)].sort_values(by=['avg_score'])

#df_zebra_low_score

image_list = df_zebra_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Zebra",model_trained)

"""Print high score images"""

df_zebra_low_score = df_zebra[(df_zebra["avg_score"] > 0.8)].sort_values(by=['avg_score'])

image_list = df_zebra_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Zebra",model_trained)

"""# **Predict Windows Dataset**"""

windows_image_path = utils.repo_image_path('/Street windows')

windows_annos_dir = utils.repo_image_path('/Street windows/annotations')

df_windows, windows_iou = utils.pipeline('windows', windows_image_path, windows_annos_dir, 'jpg', model_trained, '.xml')

#df_windows

print(windows_iou)

df_windows_low_score = df_windows[(df_windows["avg_score"] < 0.5)].sort_values(by=['avg_score'])

#df_windows_low_score

image_list = df_windows_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Street windows",model_trained)

# try bad example
window_example = utils.repo_image_path('/Street windows/000003.jpg')
utils.predict_plot_image(window_example,model_trained)

# Try good axample
window_example2 = utils.repo_image_path('/Street windows/000004.jpg')
utils.predict_plot_image(window_example2,model_trained)

"""# **Predict Kangaroos Dataset**"""

kangaroos_image_path = utils.repo_image_path('/Kangaroos')

kangaroos_annos_dir = utils.repo_image_path('/Kangaroos/annots')

df_kangaroos, kangaroos_iou = utils.pipeline('kangaroos', kangaroos_image_path, kangaroos_annos_dir,'jpg', model_trained, '.xml')

#df_kangaroos

print(kangaroos_iou)

df_kangaroos_low_score = df_kangaroos[(df_kangaroos["avg_score"] < 0.5)].sort_values(by=['avg_score'])

image_list = df_kangaroos_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Kangaroos",model_trained)

"""# **Predict Face mask Dataset**"""

face_mask_image_path = utils.repo_image_path('/Face mask dataset')

face_mask_annos_dir = utils.repo_image_path('/Face mask dataset/annotations')

df_face_mask, face_mask_iou = utils.pipeline('face_mask', face_mask_image_path, face_mask_annos_dir, 'jpg', model_trained, '.xml')

print(face_mask_iou)

df_face_mask_low_score = df_face_mask[(df_face_mask["avg_score"] < 0.5)].sort_values(by=['avg_score'])

image_list = df_face_mask_low_score.index.values.tolist()

for image in image_list:
  utils.print_image_by_dataset_and_name(image, "Face mask dataset",model_trained)

"""# **Predict B&W mask Dataset**"""
