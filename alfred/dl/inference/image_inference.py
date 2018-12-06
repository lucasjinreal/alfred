"""

Simple demo

mother method of all demos


"""
import cv2
import os
import sys


class ImageInferEngine(object):

    def __init__(self, f, is_show=True, record=False):
        """
        run on

        1. single file
        2. video file
        3. webcam
        :param f:
        """
        self.img_ext = ['png', 'jpg', 'jpeg']
        self.video_ext = ['avi', 'mp4']
        self.f = f
        self.is_show = is_show
        self.record = record

        if not f:
            self.mode = 'webcam'
        elif os.path.basename(f).split('.')[-1] in self.img_ext:
            self.mode = 'image'
        elif os.path.basename(f).split('.')[-1] in self.video_ext:
            self.mode = 'video'
        print('[Demo] in {} mode.'.format(self.mode))

    def read_image_file(self, img_f):
        """
        this method should return image read result

        :param img_f:
        :return:
        """
        if self.mode == 'image':
            raise NotImplementedError('read_image_file must be implemented when using image mode ')
        else:
            pass

    def solve_a_image(self, img):
        """
        this method must be implemented to solve a single image

        img must be a numpy array
        then return the detection or segmentation out
        :param img:
        :return:
        """
        raise NotImplementedError('solve_a_image method must be implemented')

    def vis_result(self, net_out):
        """
        this method must be implement to visualize result on image

        :param net_out:
        :return:
        """
        raise NotImplementedError('Visualize network output on image')

    def run(self):
        if self.mode == 'image':
            img = self.read_image_file(self.f)
            res = self.solve_a_image(img)
            res_img = self.vis_result(res)
            if self.is_show:
                cv2.imshow('result', res_img)
                cv2.waitKey(1)
            return res, res_img
        elif self.mode == 'video' or self.mode == 'webcam':

            cap = cv2.VideoCapture(self.f)
            if self.record and self.mode == 'video':
                fps = cap.get(cv2.CAP_PROP_FPS)
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                save_f = os.path.join(os.path.dirname(self.f), 'result_' + os.path.basename(self.f))
                print('Result video will record into {}'.format(save_f))
                video_writer = cv2.VideoWriter(save_f, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            while cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    res = self.solve_a_image(frame)
                    res_img = self.vis_result(res)
                    if self.record and self.mode == 'video':
                        video_writer.write(res_img)

                    if self.is_show:
                        cv2.imshow('result', res_img)
                        cv2.waitKey(0)
                    return res, res_img
                else:
                    print('Done')
                    exit(0)






