"""
gather all voc labels from Annotations root folder
which contains all xml annotations

"""

"""

gather the label from Annotations
"""
import os
import pickle
import os.path
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import glob



def gather_labels(anno_dir):
    all_labels = glob.glob(os.path.join(anno_dir, '*.xml'))
    all_names = []
    for label in all_labels:
        print('parsing {}'.format(label))
        root = ET.parse(label).getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text.lower().strip()
            print(name)
            if name not in all_names:
                all_names.append(name)
    print('Done. summary...')
    print('all {} classes.'.format(len(all_names)))
    print(all_names)


