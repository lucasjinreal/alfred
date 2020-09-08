import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import glob

classes = ['traffic_light', 'red_stop_left', 'green_go', 'red_stop', 'green_number', 'green_go_left',
           'red_number', 'yellow_number', 'yellow_warning', 'green_go_u-turn', 'yellow_warning_left', 'green_go_straight']

# soft link your VOC2018 under here
ann_d = sys.argv[1]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert_annotation(xml_f, target_dir):
    f_name = os.path.basename(xml_f).split('.')[0] + '.txt'
    out_file = open(os.path.join(target_dir, f_name), 'w')

    tree = ET.parse(xml_f)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

ann_name = os.path.basename(os.path.abspath(ann_d))
labels_target = os.path.join(os.path.dirname(ann_d), '{}_yolo'.format(ann_name))
print('labels dir to save: {}'.format(labels_target))
if not os.path.exists(labels_target):
    os.makedirs(labels_target)

xmls = glob.glob(os.path.join(ann_d, '*.xml'))
for xml in xmls:
    convert_annotation(xml, labels_target)

print('done.')
