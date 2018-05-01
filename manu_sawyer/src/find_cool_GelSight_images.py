#!/usr/bin/env python

import numpy as np
import h5py
from tqdm import tqdm
import cv2
import grip_and_record.getch
import matplotlib.pyplot as plt
import tensorflow_model_is_gripping.aolib.img as ig
import time

def find_cool_images(path, filenames):
    """

    :param filenames: one string or a list of strings
    :return:
    """
    if isinstance(filenames, basestring):
        filename = [filenames]  # only a string is convert to a list

    for filename in filenames:
        path_to_file = path + filename

        print('Opening file: %s' % path_to_file)
        f = h5py.File(path_to_file, "r")

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
        plt.ion()
        plt.show()

        print("Is it a cool image? [y/n]")
        done = False
        cool = False
        while not done:
            c = grip_and_record.getch.getch()
            if c:
                if c in ['n']:
                    cool = False
                    done = True
                elif c in ['y']:
                    cool = True
                    done = True

        if cool:
            path_cool = "/home/manu/ros_ws/src/manu_research/data/cool_images/"
            file = open(path_cool + filename[:-5] + '_GelSightA.jpeg', 'w')
            file.write(str(compressed_GelSightA))
            file.close()
            time.sleep(0.5)
            file = open(path_cool + filename[:-5] + '_GelSightB.jpeg', 'w')
            file.write(str(compressed_GelSightB))
            file.close()

        plt.close()


if __name__ == '__main__':
    directory = "/media/data_disk/dataset_manu/ver2/2017-06-22/"
    import os

    list_filenames = []
    for file in os.listdir(directory):
        if file.endswith(".hdf5"):
            list_filenames.append(file)
    list_filenames = sorted(list_filenames)
    print(list_filenames)

    path = "/media/data_disk/dataset_manu/ver2/2017-06-22/"
    find_cool_images(path, list_filenames[4:])
