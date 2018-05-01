#!/usr/bin/env python


# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm

import tensorflow_model_is_gripping.aolib.img as ig, tensorflow_model_is_gripping.aolib.util as ut


def compress_data(namefile):

    def compress_im(im):
        s = ig.compress(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "jpeg")
        # assert(np.all(ig.uncompress(im) == im))
        return np.asarray(s)

    path, file = os.path.split(namefile)
    newnamefile = os.path.join(path, os.path.splitext(file)[0] + '_compressed.hdf5')
    newfile = h5py.File(newnamefile, "w")

    print('New file %s' % newnamefile)

    f = h5py.File(namefile, "r")

    n_steps = f['n_steps'].value
    print('Number of steps: %d' % n_steps)

    # newfile.create_dataset("frequency", data=f['frequency'].value)
    newfile.create_dataset("n_steps", data=f['n_steps'].value)

    for i in tqdm(range(1, n_steps)):
        id = '/step_%012d' % i

        # data.time.append(f[id + '/timestamp'].value)
        # data.robot.limb.append(f[id + '/limb'].value.flatten())
        # data.robot.gripper.append(f[id + '/gripper'].value)
        # data.kinect.A.image.append(cv2.cvtColor(f[id + '/color_image_KinectA'].value, cv2.COLOR_BGR2RGB))
        # data.kinect.A.depth.append(f[id + '/depth_image_KinectA'].value)
        # data.kinect.B.image.append(cv2.cvtColor(f[id + '/color_image_KinectB'].value, cv2.COLOR_BGR2RGB))
        # data.kinect.B.depth.append(f[id + '/depth_image_KinectB'].value)
        # data.gelsight.A.image.append(f[id + '/GelSightA_image'].value)
        # data.gelsight.B.image.append(f[id + '/GelSightB_image'].value)

        id = '/step_%012d' % i
        ut.tic('compress')
        # depth_image_KinectA = convert_depth(depth_image_KinectA)
        color_image_KinectA = compress_im(f[id + '/color_image_KinectA'].value)
        # depth_image_KinectB = convert_depth(depth_image_KinectB)
        color_image_KinectB = compress_im(f[id + '/color_image_KinectB'].value)
        GelSightA_image = compress_im(f[id + '/GelSightA_image'].value)
        GelSightB_image = compress_im(f[id + '/GelSightB_image'].value)
        ut.toc()

        # f[id + '/color_image_KinectB'] = GelSightB_image

    # Close files
    newfile.close()
    f.close()


if __name__ == '__main__':
    nameFile = '/media/data_disk/dataset_manu/2017-06-18/2017-06-18_000134.hdf5'
    # nameFile = '/home/guser/catkin_ws/src/manu_research/data/data_trial_00000001.hdf5'
    data = compress_data(namefile=nameFile)