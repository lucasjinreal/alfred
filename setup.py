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
      version='1.0.7',
      keywords=['deep learning', 'script helper', 'tools'],
      description='alfred is a deep learning scripts collection, built with many standby functions for process image '
                  'or '
                  'text',
      license='GPL',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'alfred = alfred.alfred:main'
          ]
      },

      author="Lucas Jin",
      author_email="jinfagang10@163.com",
      url='https://github.com/jinfagang/alfred',
      platforms='any',
      install_requires=['colorama', 'opencv-python', 'requests']
      )
