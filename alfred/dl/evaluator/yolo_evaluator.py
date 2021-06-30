import os
import json
import glob
from alfred.utils.log import logger
from PIL import Image
import numpy as np
import cv2
import shutil
from tqdm import tqdm


__all__ = ['YoloEvaluator']


"""

Parsing any dataset in Yolo format.
You can simply change classes names to yours.
"""

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif',
               'tiff', 'dng']  # acceptable image suffixes


class YoloEvaluator:

    def __init__(self,
                 imgs_root,
                 labels_root,
                 infer_func,
                 model=None,
                 prep_func=None,
                 posp_func=None,
                 conf_thr=0.4,
                 iou_thr=0.5):
        assert os.path.exists(imgs_root)
        assert os.path.exists(labels_root)

        self.infer_func = infer_func
        self.model_ = model
        self.prep_func = prep_func
        self.posp_func = posp_func

        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

        self._data_root = imgs_root
        logger.info('data_root: {}'.format(self._data_root))

        self.img_files = []
        self.label_files = []
        self.load_combined_imgs_and_labels(imgs_root, labels_root)

        self.hold_vis = True
        logger.info(
            'Press space to vis image, press q to skip and continue eval.')

    def load_combined_imgs_and_labels(self, imgs_root, labels_root):
        logger.info('labels root path: {}, img root: {}, pls check it they right or not.'.format(
            imgs_root, labels_root))
        self.label_files = glob.glob(os.path.join(labels_root, "*.txt"))
        for ext in img_formats:
            self.img_files.extend(
                glob.glob(os.path.join(imgs_root, '*.{}'.format(ext))))
        if len(self.img_files) != len(self.label_files):
            imgs_names = [os.path.basename(i).split('.')[0]
                          for i in self.img_files]
            labels_names = [os.path.basename(i).split(
                '.')[0] for i in self.label_files]
            valid_files = [i for i in imgs_names if i in labels_names]
            logger.info('original imgs: {}, original labels: {}, valid num: {}'.format(
                len(self.img_files), len(self.label_files), len(valid_files)))
            # labels is more than images
            self.label_files = [os.path.join(
                labels_root, i + '.txt') for i in valid_files]
            self.img_files = [os.path.join(
                imgs_root, i + '.jpg') for i in valid_files]
        self.img_files = sorted(self.img_files)
        self.label_files = sorted(self.label_files)
        logger.info('img num: {}, label num: {}'.format(
            len(self.img_files), len(self.label_files)))
        logger.info(self.img_files[89])
        logger.info(self.label_files[89])
        logger.info('Please check if images and labels are paired properly.')

    def _load_yolo_boxes_and_labels(self, lb_file, ori_w, ori_h):
        with open(lb_file, 'r') as f:
            l = np.array([x.split() for x in f.read().strip(
            ).splitlines()], dtype=np.float32)  # labels
            l = np.clip(l, 0, 1.)
        res_boxes = []
        res_labels = []
        if len(l):
            boxes = l[:, 1:]
            labels = l[:, 0]

            for i, b in enumerate(boxes):
                cx, cy, w, h = b
                cx *= ori_w
                cy *= ori_h
                w *= ori_w
                h *= ori_h
                x = cx - w/2
                y = cy - h/2
                if x < 0 or y < 0 or w <= 2 or h <= 2:
                    continue
                res_boxes.append(np.array([x, y, x+w, y+h]).astype(np.int))
                res_labels.append(labels[i])
        return res_boxes, res_labels

    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except:
            pass
        return s

    @staticmethod
    def compute_iou(rec1, rec2):
        """
        computing IoU
        :param rec1: (x0, y0, x1, y1), which reflects
                (top, left, bottom, right)
        :param rec2: (x0, y0, x1, y1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
        S_rec2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[0], rec2[0])
        right_line = min(rec1[2], rec2[2])
        top_line = max(rec1[1], rec2[1])
        bottom_line = min(rec1[3], rec2[3])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0

    def recall_precision(self, gt, pre, iou_thr=0.5):
        # recall
        recall_sum = len(gt)
        recall_cnt = 0
        precision_sum = len(pre)
        precision_cnt = 0
        for g_bbox in gt:
            for p_bbox in pre:
                iou_ = self.compute_iou(g_bbox, p_bbox)
                # print('iou_of_recall: ', iou_)
                if iou_ >= iou_thr:
                    recall_cnt += 1
                    break
        for p_bbox in pre:
            for g_bbox in gt:
                iou_ = self.compute_iou(g_bbox, p_bbox)
                # print('iou_of_precision: ', iou_)
                if iou_ >= iou_thr:
                    precision_cnt += 1
                    break

        if recall_cnt > recall_sum:
            print(" ----------------------》》》 error recall")
        if precision_cnt > precision_sum:
            print(" ----------------------》》》 error precision")
        return (recall_sum, recall_cnt, precision_sum, precision_cnt)

    def eval_precisely(self):
        recall_sum, recall_cnt, precision_sum, precision_cnt = 0., 0., 0., 0.
        mAPs = []
        mrecall = []
        pic_cnt = 0

        tbar = tqdm(self.img_files)
        for i, identity in enumerate(tbar):
            # load ground truth
            im = Image.open(identity)
            im.verify()  # PIL verify
            shape = self.exif_size(im)
            img = cv2.imread(identity)
            w = shape[0]
            h = shape[1]

            annotation = self.label_files[i]
            bboxes, labels = self._load_yolo_boxes_and_labels(annotation, w, h)
            gt_list = bboxes

            # inference
            # inp = self.prep_func(identity)
            # out = self.model_(inp)
            # out = self.posp_func(out)
            out = self.infer_func(identity)

            for b in bboxes:
                x1, y1, x2, y2 = b
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            pre_list = []
            for m_ in out:
                x1, y1, x2, y2, cls_id, conf = m_
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if conf >= self.conf_thr:
                    pre_list.append((x1, y1, x2, y2))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            pic_cnt += 1
            eval_list = self.recall_precision(
                gt_list, pre_list, iou_thr=self.iou_thr)
            # logger.info(eval_list)

            recall_sum += eval_list[0]
            recall_cnt += eval_list[1]
            precision_sum += eval_list[2]
            precision_cnt += eval_list[3]
            # print('{} {} {} {}'.format(recall_sum, recall_cnt, precision_sum, precision_cnt))
            if (recall_sum > 0) and (precision_sum > 0):
                if eval_list[2] > 0:
                    mAPs.append(eval_list[3]/eval_list[2])
                else:
                    if eval_list[0] > 0:
                        mAPs.append(0)
                if eval_list[0] > 0:
                    mrecall.append(eval_list[1]/eval_list[0])
                # print(recall_cnt,recall_sum,precision_cnt,precision_sum)

            tbar.set_postfix(recall='{:.3f}'.format(recall_cnt/recall_sum), precision='{:.3f}'.format(
                precision_cnt/precision_sum), mAP='{:.3f}'.format(np.mean(mAPs)), refresh=True)

            if self.hold_vis:
                cv2.imshow('aa', img)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    print('Pressed Q, continue eval without vis.')
                    self.hold_vis = False
                    cv2.destroyAllWindows()
        print("[\n{:6d}] conf_thr: {:.2f}, iou_thr: {:.2f}, recall: {:.5f}, precision: {:.5f}, mrecall: {:.3f}, map: {:.3f}".
              format(pic_cnt, self.conf_thr, self.iou_thr, recall_cnt/recall_sum, precision_cnt/precision_sum, np.mean(mrecall), np.mean(mAPs)))

    def eval(self):
        recall_sum, recall_cnt, precision_sum, precision_cnt = 0., 0., 0., 0.
        mAPs = []
        mrecall = []
        pic_cnt = 0

        for i, identity in enumerate(self.img_files):
            # load ground truth
            im = Image.open(identity)
            im.verify()  # PIL verify
            shape = self.exif_size(im)
            w = shape[0]
            h = shape[1]

            annotation = self.label_files[i]
            bboxes, labels = self._load_yolo_boxes_and_labels(annotation, w, h)
            gt_list = bboxes

            # inference
            inp = self.prep_func(identity)
            out = self.model_(inp)
            out = self.posp_func(out)

            pre_list = []
            for m_ in out:
                x1, y1, x2, y2, conf = m_
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if conf >= self.conf_thr:
                    pre_list.append((y1, x1, y2, x2))

            pic_cnt += 1
            eval_list = self.recall_precision(
                gt_list, pre_list, iou_thr=self.iou_thr)

            recall_sum += eval_list[0]
            recall_cnt += eval_list[1]
            precision_sum += eval_list[2]
            precision_cnt += eval_list[3]
            if (recall_sum > 0) and (precision_sum > 0):
                if eval_list[2] > 0:
                    mAPs.append(eval_list[3]/eval_list[2])
                else:
                    if eval_list[0] > 0:
                        mAPs.append(0)
                if eval_list[0] > 0:
                    mrecall.append(eval_list[1]/eval_list[0])
                # print(recall_cnt,recall_sum,precision_cnt,precision_sum)
                print("  {:6d}>  -->>conf_thr :{:.2f} , iou_thr :{:.2f} , recall :{:.5f} , precision :{:.5f}, mrecall :{:.3f}, map :{:.3f}".
                      format(pic_cnt, self.conf_thr, self.iou_thr, recall_cnt/recall_sum, precision_cnt/precision_sum, np.mean(mrecall), np.mean(map)), end='\r')
            print('all recall: {}, all precision: {}'.format(
                recall_sum, precision_sum))
