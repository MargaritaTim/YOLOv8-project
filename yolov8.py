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
from PIL import Image, ImageStat
# from google.colab.patches import cv2_imshow

import xml.etree.ElementTree as ET

import torch
import torchvision.models as models

import ultralytics
from ultralytics import YOLO

torch.manual_seed(0)

''' YOLOv8 model '''

''' 1 Create YOLOv8 model '''
locale.getpreferredencoding = lambda: "UTF-8"

# !pip install pyyaml h5py

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# model.train(data='coco128.yaml', epochs=10, imgsz=640)

''' 2 Save the trained model '''
# export the model
# model.export()  

''' 3 Load trained model & example '''
model_trained = YOLO(utils.repo_image_path('/best.torchscript'), task='detect')

# ** Example**
#image_path = utils.repo_image_path('/Kangaroos/00050.jpg')
#utils.predict_plot_image(image_path,model_trained)

# create dictionary to save iou results for all datasets
iou_dict = {}

''' Predict COCO128 Dataset '''

coco128_path = utils.repo_image_path('/coco128/image')
coco_annos_dir = utils.repo_image_path('/coco128/annotations')

df_coco, coco_iou = utils.pipeline('coco128', coco128_path, coco_annos_dir, 'jpg', model_trained)

iou_dict["coco128"] = coco_iou

''' Predict Mouse Dataset '''

mouse_path = utils.repo_image_path('/Mouse')
mouse_annos_dir = utils.repo_image_path('/Mouse/annotations')

df_mouse, mouse_iou = utils.pipeline('mouse', mouse_path, mouse_annos_dir, 'jpg', model_trained)

iou_dict["mouse"] = mouse_iou

# print images with low score
# df_mouse_low_score = df_mouse[(df_mouse["avg_score"] < 0.5)].sort_values(by=['avg_score'])
#mouse_low_score_lst = df_mouse_low_score.index.values.tolist()

#for image in mouse_low_score_lst:
#  utils.print_image_by_dataset_and_name(image, "Mouse", model_trained)

# print images with high score
#df_mouse_high_score = df_mouse[(df_mouse["avg_score"] > 0.8)].sort_values(by=['avg_score'])

#mouse_high_score_list = df_mouse_high_score.index.values.tolist()

#for image in mouse_high_score_list:
#  utils.print_image_by_dataset_and_name(image, "Mouse",model_trained)

""" Predict Zebras Dataset """

zebra_image_path = utils.repo_image_path('/Zebra')
zebra_annos_dir = utils.repo_image_path('/Zebra/annotations')

df_zebra, zebra_iou = utils.pipeline('zebra', zebra_image_path, zebra_annos_dir, 'jpg', model_trained)

iou_dict["zebra"] = zebra_iou

# print low score images
#df_zebra_low_score = df_zebra[(df_zebra["avg_score"] < 0.5)].sort_values(by=['avg_score'])
#zebra_low_score_list = df_zebra_low_score.index.values.tolist()

#for image in zebra_low_score_list:
#  utils.print_image_by_dataset_and_name(image, "Zebra",model_trained)

# print high score images
#df_zebra_low_score = df_zebra[(df_zebra["avg_score"] > 0.8)].sort_values(by=['avg_score'])
#zebra_low_score_list = df_zebra_low_score.index.values.tolist()

#for image in zebra_low_score_list:
#  utils.print_image_by_dataset_and_name(image, "Zebra",model_trained)

""" Predict Windows Dataset """

windows_image_path = utils.repo_image_path('/Street windows')
windows_annos_dir = utils.repo_image_path('/Street windows/annotations')

df_windows, windows_iou = utils.pipeline('windows', windows_image_path, windows_annos_dir, 'jpg', model_trained, '.xml', None)
#print(windows_iou)

iou_dict["windows"] = windows_iou

# print low score images
#df_windows_low_score = df_windows[(df_windows["avg_score"] < 0.5)].sort_values(by=['avg_score'])
#windows_low_score_list = df_windows_low_score.index.values.tolist()

#for image in windows_low_score_list:
#  utils.print_image_by_dataset_and_name(image, "Street windows",model_trained)

# try bad example
#window_example = utils.repo_image_path('/Street windows/000003.jpg')
#utils.predict_plot_image(window_example,model_trained)

# Try good axample
#window_example2 = utils.repo_image_path('/Street windows/000004.jpg')
#utils.predict_plot_image(window_example2,model_trained)

""" Predict Kangaroos Dataset """

kangaroos_image_path = utils.repo_image_path('/Kangaroos')
kangaroos_annos_dir = utils.repo_image_path('/Kangaroos/annots')

df_kangaroos, kangaroos_iou = utils.pipeline('kangaroos', kangaroos_image_path, kangaroos_annos_dir,'jpg', model_trained, '.xml', None)

iou_dict["kangaroos"] = kangaroos_iou

# print low score images
#df_kangaroos_low_score = df_kangaroos[(df_kangaroos["avg_score"] < 0.5)].sort_values(by=['avg_score'])
#kangaroos_low_score_list = df_kangaroos_low_score.index.values.tolist()

#for image in kangaroos_low_score_list:
#  utils.print_image_by_dataset_and_name(image, "Kangaroos",model_trained)

""" Predict Face mask Dataset """

#face_mask_image_path = utils.repo_image_path('/Face mask dataset')
#face_mask_annos_dir = utils.repo_image_path('/Face mask dataset/annotations')

#df_face_mask, face_mask_iou = utils.pipeline('face_mask', face_mask_image_path, face_mask_annos_dir, 'jpg', model_trained, '.xml')

#iou_dict["face mask"] = face_mask_iou

# print low score images
#df_face_mask_low_score = df_face_mask[(df_face_mask["avg_score"] < 0.5)].sort_values(by=['avg_score'])
#face_mask_low_score_list = df_face_mask_low_score.index.values.tolist()

#for image in face_mask_low_score_list:
#  utils.print_image_by_dataset_and_name(image, "Face mask dataset",model_trained)

""" Predict B&W Dataset """

#bw_zebra_image_path = utils.repo_image_path('/BW-Zebra')

# create folder for B&W dataset
#if not os.path.exists(bw_zebra_image_path):
#    os.makedirs(bw_zebra_image_path)

# convert images to grayscale
#for filename in os.listdir(zebra_image_path):
#    if filename.endswith(".jpg") or filename.endswith(".png"):
#        img_path = os.path.join(zebra_image_path, filename)
#        img = Image.open(img_path).convert('L')
#        img.save(os.path.join(zebra_image_path, filename))

#df_bw_zebra, bw_zebra_iou = utils.pipeline('bw_zebra', zebra_image_path, zebra_annos_dir, 'jpg', model_trained, None, "BW")

#iou_dict["bw_zebra"] = bw_zebra_iou

print(iou_dict)

""" Image properties """

#df_images['avg_score'] = df_images.apply(lambda row: sum(row['max_iou_score']) / row['num_of_annotations'], axis=1)
#df_mouse
#return_aspect_ratio(w,h)
#df_images['relative_boxes'] = df_images.apply(lambda row: boxes_abs_to_relative(row['boxes'], row['height'], row['width']), axis=1)
#df_zebra
#df_windows
#df_kangaroos