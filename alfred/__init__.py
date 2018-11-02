# -*- coding: utf-8 -*-
# file: __init__.py
# author: JinTian
# time: 05/02/2018 9:20 PM
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
"""Bring in all of the public Alfred interface into this module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order

from .modules import fusion
from .modules import scrap
from .modules import text
from .modules import vision

from .modules.fusion import fusion_utils
from .modules.vision import vis_kit