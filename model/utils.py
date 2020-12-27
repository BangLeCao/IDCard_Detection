import cv2
import os
import pandas as pd
import torch
import xml.etree.ElementTree as ET

from glob import glob
from torchvision import transforms


def default_transforms():
    return transforms.Compose([transforms.ToTensor(), normalize_transform()])

def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def read_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def reverse_normalize(image):
    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image)

def xml_to_csv(xml_folder, output_file=None):
    xml_list = []
    image_id = 0
    # Loop every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(float(box.find('xmin').text)),
                   int(float(box.find('ymin').text)), int(float(box.find('xmax').text)), int(float(box.find('ymax').text)), image_id)
            xml_list.append(row)
        
        image_id += 1

    # Save to a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df


# Checks whether a variable is a list or tuple only
def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)
