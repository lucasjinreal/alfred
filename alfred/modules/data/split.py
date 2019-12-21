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