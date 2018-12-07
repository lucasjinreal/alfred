# Alfred

*Alfred* is command line tool for deep-learning usage. if you want split an video into image frames or combine frames into a single video, then **alfred** is what you want.

![showing using alfred combine a image sequence into video](https://i.loli.net/2018/08/01/5b612d34d9872.png)



## Install

To install **alfred**, it is very simple:

```shell
sudo pip3 install alfred-py
```

And then you can do something like this in your python program:

```python
from alfred.fusion import fusion_utils
```



## Updates



- 2018-12-7:

  Now, we adding a extensible class for quickly write an image detection or segmentation demo.

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

  <div align=center><img src="https://s1.ax1x.com/2018/12/07/F1bjwd.gif"/></div>

  The repo using **alfred**: http://github.com/jinfagang/pt_enet

- 2018-11-6:

  I am so glad to announce that alfred 2.0 released！😄⛽️👏👏  Let's have a quick look what have been updated:

  ```
  # 2 new modules, fusion and vis
  from alred.fusion import fusion_utils
  ```

  For the module `fusion` contains many useful sensor fusion helper functions you may use, such as project lidar point cloud onto image.

- 2018-08-01

  Fix the video combined function not work well with sequence. Add a order algorithm to ensure video sequence right.
  also add some draw bbox functions into package.

  can be called like this:

- 2018-03-16

  Slightly update **alfred**, now we can using this tool to combine a video sequence back original video!
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