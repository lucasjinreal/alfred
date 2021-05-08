

"""

Split coco dataset

"""
import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
import os


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def split_coco(ann_f, split=0.8, remove_empty=False):
    train_f = os.path.join(os.path.dirname(ann_f), os.path.basename(
        ann_f).replace('.json', '_train.json'))
    val_f = os.path.join(os.path.dirname(ann_f), os.path.basename(
        ann_f).replace('.json', '_val.json'))
    with open(ann_f, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = ''
        if 'info' in coco.keys():
            info = coco['info']
        licenses = ''
        if 'licenses' in coco.keys():
            licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        if remove_empty:
            images_with_annotations = funcy.lmap(
                lambda a: int(a['image_id']), annotations)

            # filter out images without annotations
            images = funcy.lremove(
                lambda i: i['id'] not in images_with_annotations, images)
            print('removed {} images without annotations, all images: {}, now: {}'.format(
                number_of_images-len(images), number_of_images, len(images)
            ))
        else:
            print('all images: {}'.format(number_of_images))

        x, y = train_test_split(images, train_size=float(split))

        save_coco(train_f, info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco(val_f, info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}.".format(
            len(x), train_f, len(y), val_f))
