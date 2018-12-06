# -*- coding: utf-8 -*-
# file: setup.py
# author: JinTian
# time: 04/02/2018 12:16 PM
# Copyright 2018 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
install alfred into local bin dir.
"""
from setuptools import setup, find_packages

setup(name='alfred-py',
      version='2.3.5',
      keywords=['deep learning', 'script helper', 'tools'],
      description='''
      Alfred is a DeepLearning utility library.
      ''',
      long_description='''
      Alfred is a DeepLearning utility library. it consist of text pre-processing,
      computer vision visualization, and some sensor fusion functions. You can even
      load some deep learning base nets from alfred. Everything is just include at you 
      wish to call it.
      
      Also, you can using alfred as a console program to execute some functions, such as
      combine video, image processing, scrap image from internet etc.
      ''',
      license='GPL',
      packages=[
          'alfred',
          'alfred.dl',
          'alfred.dl.inference',
          'alfred.dl.torch',
         # 'alfred.dl.torch.data',
          'alfred.vis',
          'alfred.modules',
          'alfred.modules.scrap',
          'alfred.modules.text',
          'alfred.modules.vision',
          'alfred.modules',
          'alfred.fusion',
          'alfred.vis.image',
          'alfred.vis.point_cloud'],
      # package_dir={'alfred': 'alfred'},
      entry_points={
          'console_scripts': [
              'alfred = alfred.alfred:main'
          ]
      },

      author="Lucas Jin",
      author_email="jinfagang19@163.com",
      url='https://github.com/jinfagang/alfred',
      platforms='any',
      install_requires=['colorama', 'opencv-contrib-python', 'requests', 'numpy', 'future']
      )
