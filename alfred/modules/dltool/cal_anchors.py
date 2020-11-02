import numpy as np
from glob import glob
import os
import xml.etree.ElementTree as ET


class KmeansYolo:

    def __init__(self, ann_dir, cluster_number, ann_format='voc'):
        assert ann_format.lower() == 'voc' or ann_format.lower() == 'yolo', 'only support voc or yolo format now.'
        self.cluster_number = cluster_number
        self.ann_dir = ann_dir
        self.ann_format = ann_format.lower()
        print('KMeans clustering on: {} clusters.'.format(cluster_number))

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def _load_yolo_boxes(self, ann_files):
        assert isinstance(ann_files, list), 'ann_files must be a list'
        dataSet = []
        for f in ann_files:
            lines = open(f, 'r').readlines()
            infos = [i.strip() for i in lines]
            for line in infos:
                width = int(line.split(" ")[3])
                height = int(line.split(",")[4])
                dataSet.append([width, height])
        return dataSet
    
    def _load_voc_boxes(self, ann_files):
        assert isinstance(ann_files, list), 'ann_files must be a list'
        dataSet = []
        for f in ann_files:
            tree = ET.parse(f)
            root = tree.getroot()
            # size = root.find('size')
            # im_w = int(size.find('width').text)
            # im_h = int(size.find('height').text)

            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
               
                w = float(xmlbox.find('xmax').text) - float(xmlbox.find('xmin').text)
                h = float(xmlbox.find('ymax').text) - float(xmlbox.find('ymin').text)
                dataSet.append([w, h])
        return dataSet

    def txt2boxes(self):
        files = None
        dataSet = None
        if self.ann_format == 'yolo':
            files = glob(os.path.join(self.ann_dir, '*.txt'))
            dataSet = self._load_yolo_boxes(files)
            print('[WARN] yolo format anchor calculation **only** support relative anchor (same as YoloV5 format)' +
            'if you want using it in YoloV3, you gonna need x image wh.')
        elif self.ann_format == 'voc':
            files = glob(os.path.join(self.ann_dir, '*.xml'))
            dataSet = self._load_voc_boxes(files)
        print('annotation format: {}, all: {} annos.'.format(self.ann_format, len(files)))
        print('all boxes num: {}'.format(len(dataSet)))
        result = np.array(dataSet)
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        print('clustering for anchors....')
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    # Row format: image_file_path box1 box2 ... boxN;
    # Box format: x_min,y_min,x_max,y_max,class_id (no space).
    kmeans = KmeansYolo(cluster_number, filename)
    kmeans.txt2clusters()