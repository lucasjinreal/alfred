"""
This file using for extracting faces of all images

"""
import glob
import dlib
import os
import cv2


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
                save_face_f = os.path.join(os.path.dirname(img_f), os.path.basename(img_f) + '_face_{}.png'.format(i))

                # get the face crop
                x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                print('x1: {}, y1: {}, x2: {}, y2: {}'.format(x1, y1, x2, y2))
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                img = cv2.imshow('rr', img)
                cv2.imshow('tt', img)
                cv2.waitKey(0)

                cropped_face = img[x1: x2, y1: y2]
                print('cropped size: ', cropped_face.shape)
                cv2.imwrite(save_face_f, cropped_face)
        print('Done!')

