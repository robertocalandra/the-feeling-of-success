#!/usr/bin/env python

import numpy as np
import h5py
from tqdm import tqdm
import cv2
import grip_and_record.getch
import matplotlib.pyplot as plt
import tensorflow_model_is_gripping.aolib.img as ig


def set_label(path, filenames):
    """

    :param filenames: one string or a list of strings
    :return:
    """
    if isinstance(filenames, basestring):
        filename = [filenames]  # only a string is convert to a list

    for filename in filenames:
        path_to_file = path + filename

        print('Opening file: %s' % path_to_file)
        f = h5py.File(path_to_file, "r+")


        times = np.array(f['/timestamp'].value)

        time_pre_grasping = float(f["time_pre_grasping"].value)
        index_time_pre_grasping = np.argmin(
            np.abs(times - time_pre_grasping))  # Find index corresponding to timestamp

        GelSightA_image_time_pre_grasping = np.array(
            ig.uncompress(f['/GelSightA_image'].value[index_time_pre_grasping]))
        GelSightB_image_time_pre_grasping = np.array(
            ig.uncompress(f['/GelSightB_image'].value[index_time_pre_grasping]))

        KinectA_color_time_pre_grasping = np.array(
            ig.uncompress(f['/color_image_KinectA'].value[index_time_pre_grasping]))
        KinectB_color_time_pre_grasping = np.array(
            ig.uncompress(f['/color_image_KinectB'].value[index_time_pre_grasping]))

        plt.imshow(KinectA_color_time_pre_grasping)
        plt.axis('off')
        plt.draw()
        plt.ion()
        plt.show()

        str(f['object_name'].value)

        print("Does the name of the object correspond to the one printed below? [y/n]")
        print(str(f['object_name'].value))
        done = False
        while not done:
            c = grip_and_record.getch.getch()
            if c:
                if c in ['n']:
                    name = input("Enter the correct name: ")
                    data = f['object_name']
                    data[...] = np.asarray([name])
                    done = True
                elif c in ['y']:
                    done = True

        f.flush()
        f.close()
        plt.close()


if __name__ == '__main__':
    directory = "/home/manu/ros_ws/src/manu_research/data/"
    import os

    list_filenames = []
    for file in os.listdir(directory):
        if file.endswith(".hdf5"):
            list_filenames.append(file)
    list_filenames = sorted(list_filenames)
    print(list_filenames)

    path = directory
    set_label(path, list_filenames[-30:])
