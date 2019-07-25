import os
import glob
import numpy as np

save_dir = './ImageSets/Main'
os.makedirs(save_dir, exist_ok=True)

all_imgs = [os.path.basename(i) for i in glob.glob('./JPEGImages/*.jpg')]
all_imgs += [os.path.basename(i) for i in glob.glob('./JPEGImages/*.png')]

suppose_labels = [os.path.join('./Annotations', i.split('.')[0] + '.xml') for i in all_imgs]

reasonable_imgs = []
for i in range(len(suppose_labels)):
    # print(suppose_labels[i])
    if os.path.exists(suppose_labels[i]):
        reasonable_imgs.append(all_imgs[i])
    else:
        print('find one image does not have label: {}'.format(all_imgs[i]))

# some image may not have labels, filter it..
ratio = 0.9
print('Found {} images, spliting ratio is 0.9, original image length is: {}, some does not have label, drop it.'.format(len(reasonable_imgs), len(all_imgs)))


all_imgs = [i+'\n' for i in reasonable_imgs]
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