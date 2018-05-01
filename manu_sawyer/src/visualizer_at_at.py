#!/usr/bin/env python

import numpy as np
import h5py
from tqdm import tqdm
import cv2
import grip_and_record.getch
import matplotlib.pyplot as plt
import tensorflow_model_is_gripping.aolib.img as ig
import time

def visualize(filename):
    """

    :param filenames: one string
    :return:
    """

    print('Opening file: %s' % filename)
    f = h5py.File(filename, "r")

    times = np.array(f['/timestamp'].value)

    time_at_grasping = float(f["time_at_grasping"].value)
    index_time_at_grasping = np.argmin(
        np.abs(times - time_at_grasping))  # Find index corresponding to timestamp

    compressed_kinectA = f['/color_image_KinectA'].value[index_time_at_grasping]
    KinectA_color_time_at_grasping = np.array(ig.uncompress(compressed_kinectA))

    compressed_kinectB = f['/color_image_KinectB'].value[index_time_at_grasping]
    KinectB_color_time_at_grasping = np.array(ig.uncompress(compressed_kinectB))

    compressed_GelSightA = f['/GelSightA_image'].value[index_time_at_grasping]
    GelSightA_image_time_at_grasping = np.array(ig.uncompress(compressed_GelSightA))

    compressed_GelSightB = f['/GelSightB_image'].value[index_time_at_grasping]
    GelSightB_image_time_at_grasping = np.array(ig.uncompress(compressed_GelSightB))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(KinectA_color_time_at_grasping)
    ax2.imshow(KinectB_color_time_at_grasping)
    ax3.imshow(GelSightA_image_time_at_grasping)
    ax4.imshow(GelSightB_image_time_at_grasping)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    plt.draw()
    plt.show()


if __name__ == '__main__':
    namefile = '/home/manu/ros_ws/src/manu_research/data/2017-06-26_210256.hdf5'
    namefile = '/media/backup_disk/dataset_manu/ver2/2017-06-22/2017-06-22_235321.hdf5'
    visualize(namefile)
