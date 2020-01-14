# Alfred

*Alfred* is command line tool for deep-learning usage. if you want split an video into image frames or combine frames into a single video, then **alfred** is what you want.



## Install

To install **alfred**, it is very simple:

```shell
sudo pip3 install alfred-py
```

A glance of alfred, after you installed above package, you will have `alfred`:

```
# show VOC annotations
alfred data vocview -i JPEGImages/ -l Annotations/
# show more of data
alfred data -h

# cabinet module
alfred cab -h
# count jpg file number under current dir
alfred cab count -t jpg
```





## Updates


- 2050-: *to be continue*;

- 2020-01-14: Added cabinet module, also add some utils under data module;

- 2019-07-18: 1000 classes imagenet labelmap added. Call it from:

    ```python
    from alfred.vis.image.get_dataset_label_map import imagenet_labelmap
    
    # also, coco, voc, cityscapes labelmap were all added in
    from alfred.vis.image.get_dataset_label_map import coco_labelmap
    from alfred.vis.image.get_dataset_label_map import voc_labelmap
    from alfred.vis.image.get_dataset_label_map import cityscapes_labelmap
    ```
    
- 2019-07-13: We add a VOC check module in command line usage, you can now visualize your VOC format detection data like this:

    ```
    alfred data voc_view -i ./images -l labels/
    ```
    
    
- 2019-05-17: We adding **open3d** as a lib to visual 3d point cloud in python. Now you can do some simple preparation and visual 3d box right on lidar points and show like opencv!!

    ![](https://user-images.githubusercontent.com/21303438/57909386-44313500-78b5-11e9-8146-c74c53038c9b.png)

    ```python
    geometries = []
    pcs = np.array(points[:,:3])
    pcobj = PointCloud()
    pcobj.points = Vector3dVector(pcs)
    geometries.append(pcobj)
    # try getting 3d boxes coordinates
    for p in box3d:
        pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), 
        lines = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        colors = [[1, 0, 1] for i in range(len(lines))]
        line_set = LineSet()
        line_set.points = Vector3dVector(pts3d)
        line_set.lines = Vector2iVector(lines)
        line_set.colors = Vector3dVector(colors)
        geometries.append(line_set)
        draw_pcs_open3d(geometries)
    ```

    You can achieve this by only using **alfred-py** and **open3d**!

- 2019-05-10: A minor updates but **really useful** which we called **mute_tf**, do you want to disable tensorflow ignoring log? simply do this!!

    ```python
    from alfred.dl.tf.common import mute_tf
    mute_tf()
    import tensorflow as tf
    ```
    Then, the logging message were gone....

- 2019-05-07: Adding some protos, now you can parsing tensorflow coco labelmap by using alfred:
    ```python
    from alfred.protos.labelmap_pb2 import LabelMap
    from google.protobuf import text_format

    with open('coco.prototxt', 'r') as f:
        lm = LabelMap()
        lm = text_format.Merge(str(f.read()), lm)
        names_list = [i.display_name for i in lm.item]
        print(names_list)
    ```

- 2019-04-25: Adding KITTI fusion, now you can get projection from 3D label to image like this:
  we will also add more fusion utils such as for *nuScene* dataset.

  We providing kitti fusion kitti for convert `camera link 3d points` to image pixel, and convert `lidar link 3d points` to image pixel. Roughly going through of APIs like this:

  ```python
  # convert lidar prediction to image pixel
  from alfred.fusion.kitti_fusion import LidarCamCalibData, \
      load_pc_from_file, lidar_pts_to_cam0_frame, lidar_pt_to_cam0_frame
  from alfred.fusion.common import draw_3d_box, compute_3d_box_lidar_coords

  # consit of prediction of lidar
  # which is x,y,z,h,w,l,rotation_y
  res = [[4.481686, 5.147319, -1.0229858, 1.5728549, 3.646751, 1.5121397, 1.5486346],
         [-2.5172017, 5.0262384, -1.0679419, 1.6241353, 4.0445814, 1.4938312, 1.620804],
         [1.1783253, -2.9209857, -0.9852259, 1.5852798, 3.7360613, 1.4671413, 1.5811548]]

  for p in res:
      xyz = np.array([p[: 3]])
      c2d = lidar_pt_to_cam0_frame(xyz, frame_calib)
      if c2d is not None:
          cv2.circle(img, (int(c2d[0]), int(c2d[1])), 3, (0, 255, 255), -1)
      hwl = np.array([p[3: 6]])
      r_y = [p[6]]
      pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)

      pts2d = []
      for pt in pts3d[0]:
          coords = lidar_pt_to_cam0_frame(pt, frame_calib)
          if coords is not None:
              pts2d.append(coords[:2])
      pts2d = np.array(pts2d)
      draw_3d_box(pts2d, img)
  ```

  And you can see something like this:

  **note**:

  `compute_3d_box_lidar_coords` for lidar prediction, `compute_3d_box_cam_coords` for KITTI label, **cause KITTI label is based on camera coordinates!**.
  <p align="center">
  <img src="https://s2.ax1x.com/2019/04/24/EVrU0O.md.png" />
  </p>





- 2019-01-25: We just adding network visualization tool for **pytorch** now!! How does it look? Simply print out *every layer network with output shape*,  I believe this is really helpful for people to visualize their models!

  ```
  ➜  mask_yolo3 git:(master) ✗ python3 tests.py
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1         [-1, 64, 224, 224]           1,792
                ReLU-2         [-1, 64, 224, 224]               0
                .........
             Linear-35                 [-1, 4096]      16,781,312
               ReLU-36                 [-1, 4096]               0
            Dropout-37                 [-1, 4096]               0
             Linear-38                 [-1, 1000]       4,097,000
  ================================================================
  Total params: 138,357,544
  Trainable params: 138,357,544
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.19
  Forward/backward pass size (MB): 218.59
  Params size (MB): 527.79
  Estimated Total Size (MB): 746.57
  ----------------------------------------------------------------
  
  ```

  Ok, that is all. what you simply need to do is:

  ```python
  from alfred.dl.torch.model_summary import summary
  from alfred.dl.torch.common import device
  
  from torchvision.models import vgg16
  
  vgg = vgg16(pretrained=True)
  vgg.to(device)
  summary(vgg, input_size=[224, 224])
  ```

  Support you input (224, 224) image, you will got this output, or you can change any other size to see how output changes. (currently not support for 1 channel image)

- 2018-12-7: Now, we adding a extensible class for quickly write an image detection or segmentation demo.

  If you want write a demo which **do inference on an image or an video or right from webcam**, now you can do this in standared alfred way:

  ```python
  class ENetDemo(ImageInferEngine):
  
      def __init__(self, f, model_path):
          super(ENetDemo, self).__init__(f=f)
  
          self.target_size = (512, 1024)
          self.model_path = model_path
          self.num_classes = 20
  
          self.image_transform = transforms.Compose(
              [transforms.Resize(self.target_size),
               transforms.ToTensor()])
  
          self._init_model()
  
      def _init_model(self):
          self.model = ENet(self.num_classes).to(device)
          checkpoint = torch.load(self.model_path)
          self.model.load_state_dict(checkpoint['state_dict'])
          print('Model loaded!')
  
      def solve_a_image(self, img):
          images = Variable(self.image_transform(Image.fromarray(img)).to(device).unsqueeze(0))
          predictions = self.model(images)
          _, predictions = torch.max(predictions.data, 1)
          prediction = predictions.cpu().numpy()[0] - 1
          return prediction
  
      def vis_result(self, img, net_out):
          mask_color = np.asarray(label_to_color_image(net_out, 'cityscapes'), dtype=np.uint8)
          frame = cv2.resize(img, (self.target_size[1], self.target_size[0]))
          # mask_color = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]))
          res = cv2.addWeighted(frame, 0.5, mask_color, 0.7, 1)
          return res
  
  
  if __name__ == '__main__':
      v_f = ''
      enet_seg = ENetDemo(f=v_f, model_path='save/ENet_cityscapes_mine.pth')
      enet_seg.run()
  ```

  After that, you can directly inference from video. This usage can be found at git repo: 

  <p align="center"><img src="https://s1.ax1x.com/2018/12/07/F1OKLF.gif"/></p>
The repo using **alfred**: http://github.com/jinfagang/pt_enet
  
- 2018-11-6: I am so glad to announce that alfred 2.0 released！😄⛽️👏👏  Let's have a quick look what have been updated:

  ```
  # 2 new modules, fusion and vis
  from alred.fusion import fusion_utils
  ```

  For the module `fusion` contains many useful sensor fusion helper functions you may use, such as project lidar point cloud onto image.

- 2018-08-01:  Fix the video combined function not work well with sequence. Add a order algorithm to ensure video sequence right.
  also add some draw bbox functions into package.

  can be called like this:

- 2018-03-16: Slightly update **alfred**, now we can using this tool to combine a video sequence back original video!
  Simply do:

  ```shell
  # alfred binary exectuable program
  alfred vision 2video -d ./video_images
  ```


## Capable

**alfred** is both a library and a command line tool. It can do those things:

```angular2html
# extract images from video
alfred vision extract -v video.mp4
# combine image sequences into a video
alfred vision 2video -d /path/to/images
# get faces from images
alfred vision getface -d /path/contains/images/

```

Just try it out!!

## Copyright

**Alfred** build by *Lucas Jin* with ❤️， welcome star and send PR. If you got any question, you can ask me via wechat: `jintianiloveu`, this code released under MIT license.