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
        f = h5py.File(path_to_file, "r")

        n_steps = int(f['n_steps'].value)

        # times = []
        # for i in tqdm(range(1, n_steps)):
        #     id = '/step_%012d' % i
        #     times.append(f[id + '/timestamp'].value)
        # times = np.array(times)

        times = np.array(f['/timestamp'].value)

        time_pre_grasping = float(f["time_pre_grasping"].value)
        index_time_pre_grasping = np.argmin(
            np.abs(times - time_pre_grasping))  # Find index corresponding to timestamp

        GelSightA_image_time_pre_grasping = np.array(
            ig.uncompress(f['/GelSightA_image'].value[index_time_pre_grasping]))
        GelSightB_image_time_pre_grasping = np.array(
            ig.uncompress(f['/GelSightB_image'].value[index_time_pre_grasping]))

        time_post2_grasping = float(f["time_post2_grasping"].value)
        index_time_post2_grasping = np.argmin(
            np.abs(times - time_post2_grasping))  # Find index corresponding to timestamp
        KinectA_color_time_post2_grasping = np.array(
            ig.uncompress(f['/color_image_KinectA'].value[index_time_post2_grasping]))
        KinectB_color_time_post2_grasping = np.array(
            ig.uncompress(f['/color_image_KinectB'].value[index_time_post2_grasping]))
        GelSightA_image_time_post2_grasping = np.array(
            ig.uncompress(f['/GelSightA_image'].value[index_time_post2_grasping]))
        GelSightB_image_time_post2_grasping = np.array(
            ig.uncompress(f['/GelSightB_image'].value[index_time_post2_grasping]))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(KinectA_color_time_post2_grasping)
        ax2.imshow(KinectB_color_time_post2_grasping)
        ax3.imshow(GelSightA_image_time_post2_grasping)
        ax4.imshow(GelSightB_image_time_post2_grasping)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        plt.draw()
        plt.ion()
        plt.show()

        print("Is the robot holding the object in its gripper? [y/n]")
        done = False
        while not done:
            c = grip_and_record.getch.getch()
            if c:
                if c in ['n']:
                    is_gripping = False
                    done = True
                elif c in ['y']:
                    is_gripping = True
                    done = True

        # TODO: check if folder exists, if not create
        path_0 = "/home/manu/ros_ws/src/manu_research/data/"
        file = open(path_0 + 'labels/' + filename[:-4] + 'txt', 'w')
        file.write(str(is_gripping))
        file.close()

        file = h5py.File(path_0 + "for_Andrew/" + filename, "w")
        file.create_dataset("GelSightA_image_pre_gripping", data=GelSightA_image_time_pre_grasping)
        file.create_dataset("GelSightB_image_pre_gripping", data=GelSightB_image_time_pre_grasping)
        file.create_dataset("GelSightA_image_post_gripping", data=GelSightA_image_time_post2_grasping)
        file.create_dataset("GelSightB_image_post_gripping", data=GelSightB_image_time_post2_grasping)
        file.create_dataset("is_gripping", data=np.asarray([is_gripping]))
        file.close()

        f.close()

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

    path = directory
    set_label(path, list_filenames)
