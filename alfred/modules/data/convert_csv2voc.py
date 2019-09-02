"""
Convert a CSV labeling dataset to VOC format

CARDS_COURTYARD_B_T_frame_0011.jpg,1280,720,yourleft,647,453,824,551
CARDS_COURTYARD_B_T_frame_0011.jpg,1280,720,yourright,515,431,622,543

assuming images and csv under same folder.

"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from lxml.etree import Element, SubElement, tostring, ElementTree, tostring




def convert_one_csv_to_xml(csv_f, img_f):
    if os.path.exists(csv_f):
        csv_anno = np.loadtxt(csv_f)
        if len(csv_anno.shape) < 2 and csv_anno.shape[0] != 0:
            csv_anno = np.expand_dims(csv_anno, axis=0)
        target_path = os.path.join(os.path.dirname(csv_f), os.path.basename(csv_f).split('.')[0]+'.xml')
        # convert xml 
        if os.path.exists(img_f):
            im = Image.open(img_f)
            width = im.size[0]
            height = im.size[1]
            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'images'
            node_filename = SubElement(node_root, 'filename')
            node_filename.text = os.path.basename(img_f)
            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = str(width)
            node_height = SubElement(node_size, 'height')
            node_height.text = str(height)
            node_depth = SubElement(node_size, 'depth')
            node_depth.text = '3'
            
            for item in csv_anno:
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = label_map[item[0]]
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(int(item[1]*width))
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(int(item[1]*height))
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(int(item[2]*width))
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(int(item[3]*height))
            f = open(target_path, 'wb')
            f.write(tostring(node_root, pretty_print=True))
            f.close()
        else:
            print('image: {} not exist.'.format(img_f))
    else:
        print('!! {} not exist.'.format(csv_f))