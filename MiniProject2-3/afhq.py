import os, shutil
cat_original_dataset_dir = r'./afhq/train/cat'
dog_original_dataset_dir = r'./afhq/train/dog'
base_dir = './afhq/afhq'

train_cats_dir = os.path.join(base_dir, 'trainA')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(base_dir, 'trainB')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

test_cats_dir = os.path.join(base_dir, 'testA')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(base_dir, 'testB')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)
for filename in os.walk(cat_original_dataset_dir):
    cat_fnames = filename[2]
for filename in os.walk(dog_original_dataset_dir):
    dog_fnames = filename[2]
for i in range(1000):
#for fname in fnames:
    fname = cat_fnames[i]
    src = os.path.join(cat_original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
    fname = dog_fnames[i]
    src = os.path.join(dog_original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
for i in range(1000,1500):
#for fname in fnames:
    fname = cat_fnames[i]
    src = os.path.join(cat_original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    fname = dog_fnames[i]
    src = os.path.join(dog_original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
