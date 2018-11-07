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

  

## Copyright

**Alfred** build by *Lucas Jin* with ❤️， welcome star and send PR. If you got any question, you can ask me via wechat: `jintianiloveu`, this code released under MIT license.