#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import split_folders
from pathlib import Path

def folders_in_path(path):
    if not Path.is_dir(path):
        raise ValueError('argument is not a directory')
    yield from filter(Path.is_dir, path.iterdir())

def folders_in_depth(path, depth):
    if 0 > depth:
        raise ValueError('depth is smaller than 0')
    if 0 == depth:
        yield from folders_in_path(path)
    else:
        for folder in folders_in_path(path):
            yield from folders_in_depth(folder, depth-1)

def files_in_path(path):
    if not Path.is_dir(path):
        raise ValueError('arg is not a folder')
    yield from filter(Path.is_file, path.iterdir())

if __name__ == '__main__':
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    split_folders.ratio('input', output='gemstones_data', seed=1337, ratio=(.9, .1))
