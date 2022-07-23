#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""
convert labelone to voc format

<annotation>
  <folder>VOC2007</folder>
  <filename>000002.jpg</filename>    //文件名  
  <size>                            //图像尺寸（长宽以及通道数
    <width>335</width>
    <height>500</height>
    <depth>3</depth>
  </size>
  <object>        //检测到的物体
    <name>cat</name>    //物体类别
    <pose>Unspecified</pose>    //拍摄角度
    <truncated>0</truncated>    //是否被截断（0表示完整
    <difficult>0</difficult>    //目标是否难以识别（0表示容易识别）
    <bndbox>                    //bounding-box（包含左下角和右上角xy坐标）
      <xmin>139</xmin>
      <ymin>200</ymin>
      <xmax>207</xmax>
      <ymax>301</ymax>
    </bndbox>
  </object>
</annotation>


"""
import os
import json
import glob
import sys
from PIL import Image

try:
  from lxml.etree import Element, SubElement, tostring, ElementTree, tostring
  from lxml import etree
except ImportError:
  pass


def convert_one(a):
    os.makedirs(os.path.dirname(a)+'_voc', exist_ok=True)
    d = json.load(open(a))
    print(d)

    target_path = os.path.join(os.path.dirname(a)+'_voc', os.path.basename(a).split('.')[0]+'.xml')
    img_path = os.path.join('images', d['imagePath'])
    # convert xml 
    if os.path.exists(img_path):
        im = Image.open(img_path)
        width = im.size[0]
        height = im.size[1]
        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'images'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = d['imagePath']
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'
        
        for item in d['shapes']:
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = item['label']
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(item['points'][1][0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(item['points'][1][1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(item['points'][3][0])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(item['points'][3][1])
        f = open(target_path, 'wb')
        f.write(etree.tostring(node_root, pretty_print=True))
        f.close()
    else:
        print('xxx {} annotations according image: {} not exist.'.format(a, img_path))



def run():
    all_json_files = glob.glob(os.path.join(sys.argv[1], '*.json'))
    print(len(all_json_files))
    for i in all_json_files:
        convert_one(i)
    print('done!')



if __name__ == "__main__":
    run()