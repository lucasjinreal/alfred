"""
This file using for extracting faces of all images

"""
import glob
try:
    import dlib
except ImportError:
    print('You have not installed dlib, install from https://github.com/davisking/dlib')
    print('see you later.')
    exit(0)
import os
import cv2
import numpy as np


class FaceExtractor(object):

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def get_faces(self, img_d):
        """
        get all faces from img_d
        :param img_d:
        :return:
        """

        all_images = []
        for e in ['png', 'jpg', 'jpeg']:
            all_images.extend(glob.glob(os.path.join(img_d, '*.{}'.format(e))))
        print('Found all {} images under {}'.format(len(all_images), img_d))
        for img_f in all_images:
            img = cv2.imread(img_f, cv2.COLOR_BGR2RGB)

            dets = self.detector(img, 1)
            print('=> get {} faces in {}'.format(len(dets), img_f))
            print('=> saving faces...')
            for i, d in enumerate(dets):
                save_face_f = os.path.join(os.path.dirname(img_f), os.path.basename(img_f).split('.')[0]
                                           + '_face_{}.png'.format(i))

                # get the face crop
                x = int(d.left())
                y = int(d.top())
                w = int(d.width())
                h = int(d.height())

                face_patch = np.array(img)[y: y + h, x: x + w, 0:3]
                # print(face_patch.shape)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # cv2.imshow('tt', img)
                # cv2.waitKey(0)
                cv2.imwrite(save_face_f, face_patch)
        print('Done!')
        # cv2.waitKey(0)

