

"""

Converting standared Yolo format to VOC format
here

"""


# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
import sys
import glob


CLASS_MAPPING = {
    '0': 'name'
    # Add your remaining classes here.
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(file_prefix, width, height, voc_labels, des_dir):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(des_dir, file_prefix))


def read_file(file_path, des_dir, img_dir, classes_names):
    file_prefix = os.path.basename(file_path).split(".txt")[0]
    image_file_name = "{}.jpg".format(file_prefix)
    img_p = os.path.join(img_dir, image_file_name)
    assert os.path.exists(
        img_p), 'make sure all images under: {}'.format(img_dir)
    img = Image.open(img_p)

    w, h = img.size
    with open(file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            # voc.append(CLASS_MAPPING.get(data[0]))
            if classes_names:
                voc.append(classes_names[int(data[0])])
            else:
                voc.append(data[0])
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            cx = float(data[1]) * w
            cy = float(data[2]) * h
            x = cx - bbox_width/2
            y = cy - bbox_height/2
            voc.append(x)
            voc.append(y)
            voc.append(x + (bbox_width))
            voc.append(y + (bbox_height))
            voc_labels.append(voc)
        create_file(file_prefix, w, h, voc_labels, des_dir)
    print("Processing complete for file: {}".format(file_path))


def yolo2voc(img_dir, txt_dir, class_txt):
    # TODO: need apply a map to mapping yolo label back
    classes_names = None
    if class_txt:
        classes_names = [i.strip() for i in open(class_txt, 'r').readlines()]
    des_d = 'Annotations_yolo_converted_for_voc'
    des_d = os.path.join(os.path.dirname(txt_dir.rstrip('/')), des_d)
    print('VOC Annotations will saved into: {}'.format(des_d))
    os.makedirs(des_d, exist_ok=True)
    txts = glob.glob(os.path.join(txt_dir.rstrip('/'), '*.txt'))
    print('found {} text files in yolo format.'.format(len(txts)))
    for filename in txts:
        read_file(filename, des_d, img_dir, classes_names)


if __name__ == "__main__":
    yolo2voc(sys.argv[1], sys.argv[2], None)
