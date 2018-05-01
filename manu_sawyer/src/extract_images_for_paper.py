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

        time_post2_grasping = float(f["time_at_grasping"].value)
        index_time_post2_grasping = np.argmin(
            np.abs(times - time_post2_grasping))  # Find index corresponding to timestamp

        compressed_kinectA = f['/color_image_KinectA'].value[index_time_post2_grasping]
        KinectA_color_time_post2_grasping = np.array(ig.uncompress(compressed_kinectA))

        compressed_kinectB = f['/color_image_KinectB'].value[index_time_post2_grasping]
        KinectB_color_time_post2_grasping = np.array(ig.uncompress(compressed_kinectB))

        kinectA_depth = f['/depth_image_KinectA'].value[index_time_post2_grasping]
        KinectA_depth_time_post2_grasping = np.array(kinectA_depth).squeeze()

        kinectB_depth = f['/depth_image_KinectB'].value[index_time_post2_grasping]
        KinectB_depth_time_post2_grasping = np.array(kinectB_depth).squeeze()

        compressed_GelSightA = f['/GelSightA_image'].value[index_time_post2_grasping]
        GelSightA_image_time_post2_grasping = np.array(ig.uncompress(compressed_GelSightA))

        compressed_GelSightB = f['/GelSightB_image'].value[index_time_post2_grasping]
        GelSightB_image_time_post2_grasping = np.array(ig.uncompress(compressed_GelSightB))

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # ax1.imshow(KinectA_color_time_at_grasping)
        # ax2.imshow(KinectB_color_time_at_grasping)
        # ax3.imshow(GelSightA_image_time_at_grasping)
        # ax4.imshow(GelSightB_image_time_at_grasping)
        # ax1.axis('off')
        # ax2.axis('off')
        # ax3.axis('off')
        # ax4.axis('off')
        # plt.draw()
        # plt.ion()
        # plt.show()
        #
        # print("Is it a cool image? [y/n]")
        # done = False
        # cool = False
        # while not done:
        #     c = grip_and_record.getch.getch()
        #     if c:
        #         if c in ['n']:
        #             cool = False
        #             done = True
        #         elif c in ['y']:
        #             cool = True
        #             done = True


        path_cool = "/home/manu/ros_ws/src/manu_research/data/test_figure/"
        file = open(path_cool + filename[:-5] + '_GelSightA_at.jpeg', 'w')
        file.write(str(compressed_GelSightA))
        file.close()
        time.sleep(0.5)
        file = open(path_cool + filename[:-5] + '_GelSightB_at.jpeg', 'w')
        file.write(str(compressed_GelSightB))
        file.close()
        time.sleep(0.5)
        file = open(path_cool + filename[:-5] + '_kinectA_at.jpeg', 'w')
        file.write(str(compressed_kinectA))
        file.close()
        time.sleep(0.5)
        file = open(path_cool + filename[:-5] + '_kinectB_at.jpeg', 'w')
        file.write(str(compressed_kinectB))
        file.close()

        fig = plt.figure()
        print(KinectA_depth_time_post2_grasping.shape)
        KinectA_depth_time_post2_grasping = 1024*(KinectA_depth_time_post2_grasping>1024) + KinectA_depth_time_post2_grasping*(KinectA_depth_time_post2_grasping<1024)
        plt.imshow(KinectA_depth_time_post2_grasping, cmap='gray')
        plt.show()
        import scipyplot as spp
        spp.save2file(fig=fig, nameFile='depth_image_KinectA_at')

        fig = plt.figure()
        KinectB_depth_time_post2_grasping = 1024*(KinectB_depth_time_post2_grasping>1024) + KinectB_depth_time_post2_grasping*(KinectB_depth_time_post2_grasping<1024)
        plt.imshow(KinectB_depth_time_post2_grasping, cmap='gray')
        plt.show()
        spp.save2file(fig=fig, nameFile='depth_image_KinectB_at')

if __name__ == '__main__':
    directory = "/home/manu/ros_ws/src/manu_research/data/test_figure/"
    import os

    list_filenames = []
    for file in os.listdir(directory):
        if file.endswith(".hdf5"):
            list_filenames.append(file)
    list_filenames = sorted(list_filenames)
    print(list_filenames)

    path = "/home/manu/ros_ws/src/manu_research/data/test_figure/"
    find_cool_images(path, list_filenames)
