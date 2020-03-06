"""

stack images in matrix style

"""
import cv2
import numpy as np
from alfred.utils.log import logger as logging



def check_shape_resize_if_possible(imgs):
    shapes = [i.shape for i in imgs]
    if len(set(shapes))==1:
        return imgs
    else:
        logging.info('detected images shape not equal, resize to the first shape...')
        imgs = [cv2.resize(i, (shapes[0][1], shapes[0][0])) for i in imgs]
        return imgs



def stack_imgs(imgs_list, dim2d):
    """
    send a list of images
    then using dim2d to stack it

    for example:
        a.png
        b.png
        c.png
        d.png
    
    dim2d:
        2x2
    """
    a = int(dim2d.split('x')[0])
    b = int(dim2d.split('x')[1])
    if len(imgs_list) % a != 0 or len(imgs_list) % b:
        logging.info('dim2d {} is not applicable for {} images.'.format(dim2d, len(imgs_list)))
        exit(0)
    elif len(imgs_list) != a*b:
        logging.error('len imgs not equal to: axb={}'.format(a*b))
        exit(0)
    else:
        imgs_list = [cv2.imread(i) for i in imgs_list]
        all_raws = []
        # 2x1 bug?
        for ri in range(a):
            one_raw = []
            for ci in range(b):
                one_raw.append(imgs_list[ri*b + ci])
                logging.info('stacking row: {}, with len: {}'.format(ri, len(one_raw)))
            imgs = check_shape_resize_if_possible(one_raw)
            img_a = np.hstack(imgs)
            all_raws.append(img_a)
        all_raws = check_shape_resize_if_possible(all_raws)
        final_img = np.vstack(all_raws)
        logging.info('final combined img shape: {}'.format(final_img.shape))
        cv2.imwrite('stacked_img.jpg', final_img)
        logging.info('done.')

