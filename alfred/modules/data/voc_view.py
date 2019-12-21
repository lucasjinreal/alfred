"""

this tool helps viewing VOC format data


"""
import os
import sys
import cv2
import xml.etree.ElementTree as ET
from glob import glob
import os
import sys
import cv2
from alfred.utils.log import logger as logging


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get(root, name):
    vars = root.findall(name)
    return vars


def vis_voc(img_root, label_root):
    logging.info('img root: {}, label root: {}'.format(img_root, label_root))
    # auto detection .jpg or .png images
    img_files = glob(os.path.join(img_root, '*.[jp][pn]g'))
    for img_f in img_files:
        if os.path.exists(img_f):
            img = cv2.imread(img_f)
            label_path = os.path.join(label_root, os.path.basename(img_f).split('.')[0] + '.xml')
            if os.path.exists(label_path):
                #
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in get(root, 'object'):
                    category = get_and_check(obj, 'name', 1).text
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
                    ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
                    xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
                    ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))

                    cv2.putText(img, category, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, 1)
                cv2.imshow('voc check', img)
                cv2.waitKey(0)
            else:
                logging.warning('xxxx image: {} according label: {} not found.'.format(img_f, label_path))


if __name__ == "__main__":
    vis_voc(sys.argv[1], sys.argv[2])
