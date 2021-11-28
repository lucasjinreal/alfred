# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""
install alfred into local bin dir.
"""
from setuptools import setup, find_packages
from setuptools import setup, Extension
import io
from os import path

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='alfred-py',
      version='2.9.3',
      keywords=['deep learning', 'script helper', 'tools'],
      description='Alfred is a DeepLearning utility library.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache 2.0',
      packages=[
          'alfred',
          'alfred.dl',
          'alfred.dl.inference',
          'alfred.dl.data',
          'alfred.dl.data.common',
          'alfred.dl.data.meta',
          'alfred.dl.torch',
          'alfred.dl.torch.train',
          'alfred.dl.torch.distribute',
          'alfred.dl.torch.runner',
          'alfred.dl.torch.nn',
          'alfred.dl.torch.nn.modules',
          'alfred.dl.torch.ops',
          'alfred.dl.metrics',
          'alfred.dl.tf',
          'alfred.dl.evaluator',
          'alfred.vis',
          'alfred.modules',
          'alfred.modules.scrap',
          'alfred.modules.text',
          'alfred.modules.vision',
          'alfred.modules.data',
          'alfred.modules.dltool',
          'alfred.modules.cabinet',
          'alfred.modules.cabinet.mdparse',
          'alfred.modules.cabinet.mdparse.formatters',
          'alfred.modules.cabinet.mdparse.transformers',
          'alfred.modules.cabinet.mdparse.transformers.html',
          'alfred.modules.cabinet.mdparse.transformers.md',
          'alfred.modules',
          'alfred.fusion',
          'alfred.vis.image',
          'alfred.vis.pointcloud',
          'alfred.utils',
          'alfred.siren',
          'alfred.protos',
          'alfred.deploy.tensorrt'
      ],
      # package_dir={'alfred': 'alfred'},
      entry_points={
          'console_scripts': [
              'alfred = alfred.alfred:main'
          ]
      },
      include_package_data=True,
      author="Lucas Jin",
      author_email="jinfagang19@163.com",
      url='https://github.com/jinfagang/alfred',
      platforms='any',
      install_requires=['colorama', 'requests', 'regex', 'funcy', 'pascal-voc-writer',
                        'future', 'deprecated', 'loguru', 'pyquaternion', 'lxml', 'jsons']
      )
