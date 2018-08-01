# Alfred

*Alfred* is command line tool for deep-learning usage. if you want split an video into image frames or combine frames into a single video, then **alfred** is what you want.

![](https://i.loli.net/2018/02/05/5a77dd1e89e69.png)




Currently *alfred* contains several modules:

- vision:
  1. extract: combined command for extract images from videos;
  2. gray: covert a colorful image into gray image;
  3. 2video: combine an image sequence into a video;
  4. clean: clean all un-supported or broken image from a directory;
  5. faces: extract all faces from a single image or from a images dir;

- text:
  1. clean: clean a text file, which will clean all un-use words for nlp;
  2. translate: translate a words to target language;

- scrap:
  1. image: scrap images with a query words and save into local;

# Updates

#### 2018-08-01
Fix the video combined function not work well with sequence. Add a order algorithm to ensure video sequence right.
currently newest version is *1.0.7*.

#### 2018-03-16
Slightly update **alfred**, now we can using this tool to combine a video sequence back original video!
Simply do:
```angular2html
alfred vision 2video -d ./images
```

#### Previous

*alfred* now available on pip! to install directly:

```angular2html
sudo pip3 install alfred-py
```
**Note**: pls add sudo, because alfred will add bin file into system path.

# Install
To install *alfred*, simply:

```
sudo python3 setup.py install
```
More powerful things is, you can import alfred in your project!

```
from alfred import vision
from alfred import text
```
So that you can access all alfred abilities in your own project!
