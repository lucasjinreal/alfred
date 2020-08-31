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
import os
import glob
import numpy as np

save_dir = './ImageSets/Main'
os.makedirs(save_dir, exist_ok=True)

all_imgs = [os.path.basename(i).split('.')[0]+'\n' for i in glob.glob('./JPEGImages/*.jpg')]

ratio = 0.9
print('Found {} images, spliting ratio is 0.9'.format(len(all_imgs)))

np.random.shuffle(all_imgs)
split = int(len(all_imgs) * ratio)
train_ids = all_imgs[: split]
val_ids = all_imgs[split: ]
print('{} for train, {} for validation.'.format(len(train_ids), len(val_ids)))

print('saving split..')
with open(os.path.join(save_dir, 'train.txt'), 'w') as f:
    f.writelines(train_ids)
with open(os.path.join(save_dir, 'val.txt'), 'w') as f:
    f.writelines(val_ids)
print('Done.')