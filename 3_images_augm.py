#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import random

img_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

path = 'gemstones_data/train/'
_, subf, _ = next(os.walk(path))

for i in subf:
    _, _, min_images = next(os.walk(path+i))
    random_img = random.sample(min_images, 1)[0]
    img = load_img(path+i +'//' +random_img)  # PIL image
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    count=0
    for batch in img_gen.flow(x, batch_size=1, save_to_dir= path+i, save_prefix=i, save_format='jpeg'):
        count += 1
        if count > 20:
            break  # otherwise the generator would loop indefinitely