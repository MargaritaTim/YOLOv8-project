
import os
from os import listdir

import random
import shutil
import struct

import numpy as np
import pandas as pd
import re

from matplotlib import pyplot
import cv2
from google.colab.patches import cv2_imshow

import xml.etree.ElementTree as ET

import torch
import torchvision.models as models

import locale

from ultralytics import YOLO

from google.colab import drive
drive.mount('/content/drive')

torch.manual_seed(0)


# function to predict and plot image
def predict_plot_image(image_path):
  results = model_trained(image_path)
  res_plotted = results[0].plot()
  cv2_imshow(res_plotted)


""" Functions for prediction"""

def return_bbox_masks_probs(res_lst):
    for result in res_lst:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmenation masks outputs
        probs = result.probs  # Class probabilities for classification outputs

    return boxes, masks, probs


def create_df(dataset_dir, image_type, yolo_model):
    pred_df = pd.DataFrame()

    # iterate over images
    for image in (os.listdir(dataset_dir)):
        if (image.endswith(image_type)):
            # load and prepare image
            photo_filename = dataset_dir + "/" + image
            im = cv2.imread(photo_filename)
            h, w, _ = im.shape

            # make prediction for the image
            results = model(photo_filename)

            # save bbox, masks, probabilities
            boxes, masks, _ = return_bbox_masks_probs(results)

            list_of_boxes = boxes.xyxy.tolist()

            # append to dataframe
            df = pd.DataFrame([{'name': image, 'boxes': list_of_boxes, "height": h, "width": w}])

            pred_df = pd.concat([pred_df, df], ignore_index=True)

    return pred_df


def extract_boxes(dataset_dir):
    boxes = {}

    for txt_file in (os.listdir(dataset_dir)):
        # load and prepare image
        photo_filename = dataset_dir + "/" + txt_file

        with open(photo_filename, 'r') as file:
            bbox_list = []
            for line in file.readlines():
                # Split the line into a list of words
                words = line.strip().split()

                # Extract the label and bounding box coordinates
                label = words[0]
                bbox = [float(x) for x in words[1:]]

                # Add the bounding box to the list of bounding boxes
                bbox_list.append(bbox)
            file.close()
        boxes[txt_file] = bbox_list

    return boxes


def extract_xml_boxes(dataset_dir):
    boxes = {}

    for txt_file in (os.listdir(dataset_dir)):
        # load and prepare image
        photo_filename = dataset_dir + "/" + txt_file

        with open(photo_filename, 'r') as file:
            tree = ET.parse(photo_filename)
            root = tree.getroot()

            bbox_list = []

            for neighbor in root.iter('bndbox'):
                xmin = (neighbor.find('xmin').text)
                ymin = (neighbor.find('ymin').text)
                xmax = (neighbor.find('xmax').text)
                ymax = (neighbor.find('ymax').text)

                bbox_list.append([xmin, ymin, xmax, ymax])

        boxes[txt_file] = bbox_list

    return boxes


def create_annotations_df(annotations_dir):
    anno_dict = extract_boxes(annotations_dir)
    df_anno = pd.DataFrame({'name': anno_dict.keys(), 'annotations': anno_dict.values()})

    return df_anno


def boxes_abs_to_relative(boxes, h, w):
    relative_boxes = []

    for box in boxes:
        # if box:
        xmin = float(box[0]) / w
        ymin = float(box[1]) / h
        xmax = float(box[2]) / w
        ymax = float(box[3]) / h
        relative_boxes.append([xmin, ymin, xmax, ymax])
        # else:
        #  relative_boxes.append([])

    return relative_boxes


# yolo format to relative bbox format
def yolo_to_relative(boxes):
    relative_boxes = []

    for box in boxes:
        xmin = (box[0] - box[2] / 2)
        ymin = (box[1] - box[3] / 2)
        xmax = (box[0] + box[2] / 2)
        ymax = (box[1] + box[3] / 2)
        relative_boxes.append([xmin, ymin, xmax, ymax])

    return relative_boxes


def bbox_iou(box1, box2):
    # box1 and box2 are lists with 4 elements [xmin, ymin, xmax, ymax]
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou


# function to calculate IoU between two lists of bboxes
def calculate_iou_list(pred_bboxes, ann_bboxes):
    iou_list = []

    for ann_bbox in ann_bboxes:
        if type(ann_bbox) == "<class 'float'>":
            continue
        iou_row = []
        for pred_bbox in pred_bboxes:
            iou = bbox_iou(pred_bbox, ann_bbox)
            iou_row.append(iou)
        iou_list.append(iou_row)

    return iou_list


def max_iou(list_of_iou):
    max_lst = []

    for lst in list_of_iou:
        if lst:
            max_lst.append(max(lst))
        else:
            max_lst.append(0)

    return max_lst


"""## 4 Pipeline function"""


def pipeline(dataset_name, dataset_path, annotation_path, image_format, model, annotation_foramt=None):
    df_images = create_df(dataset_path, image_format, model)

    df_images['relative_boxes'] = df_images.apply(
        lambda row: boxes_abs_to_relative(row['boxes'], row['height'], row['width']), axis=1)

    if annotation_foramt == '.xml':
        annotaions_dict = extract_xml_boxes(annotation_path)
    else:
        annotaions_dict = extract_boxes(annotation_path)
    df_annotations = pd.DataFrame({'names': annotaions_dict.keys(), 'annotations': annotaions_dict.values()})

    df_annotations['names'] = df_annotations.apply(lambda row: re.sub('txt$', 'jpg', row['names']), axis=1)
    df_annotations['names'] = df_annotations.apply(lambda row: re.sub('xml$', 'jpg', row['names']), axis=1)

    df_images = df_images.set_index('name').join(df_annotations.set_index('names'))

    # remove empty annotations
    df_images['anno_type'] = df_images.apply(lambda row: type(row['annotations']), axis=1)
    df_images = df_images.loc[df_images['anno_type'] != float]

    if dataset_name == 'coco128':
        df_images['relative_annotations'] = df_images.apply(lambda row: yolo_to_relative(row['annotations']), axis=1)
    else:
        df_images['relative_annotations'] = df_images.apply(
            lambda row: boxes_abs_to_relative(row['annotations'], row['height'], row['width']), axis=1)

    df_images['iou_score'] = df_images.apply(
        lambda row: calculate_iou_list(row['relative_boxes'], row['relative_annotations']), axis=1)

    df_images['max_iou_score'] = df_images.apply(lambda row: max_iou(row['iou_score']), axis=1)

    df_images['num_of_annotations'] = df_images.apply(lambda row: len(row['annotations']), axis=1)

    df_images['avg_score'] = df_images.apply(lambda row: sum(row['max_iou_score']) / row['num_of_annotations'], axis=1)

    total_iou = df_images['avg_score'].mean()

    return df_images, total_iou
