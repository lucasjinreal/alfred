import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys
import glob



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


def convert_annotation(xml_f, target_dir, classes_names):
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
        if cls not in classes_names or int(difficult) == 1:
            continue
        cls_id = classes_names.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')



def voc2yolo(img_dir, xml_dir, class_txt):
    classes_names = None
    if class_txt:
        classes_names = [i.strip() for i in open(class_txt, 'r').readlines()]

    labels_target = os.path.join(os.path.dirname(xml_dir.rstrip('/')), 'yolo_converted_from_voc')
    print('labels dir to save: {}'.format(labels_target))
    if not os.path.exists(labels_target):
        os.makedirs(labels_target)

    xmls = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml in xmls:
        convert_annotation(xml, labels_target, classes_names)
    print('Done!')
    print('class name order used is: ', classes_names)