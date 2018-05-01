#!/usr/bin/env python

import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
import multiprocessing
from sensor_msgs.msg import Image
import grip_and_record.locate_cylinder as locate_cylinder
import cv2


class KinectB_hd:
    def __init__(self):
        """
        Class for recording data from the KinectB
        """
        # Set up the subscribers
        self.image_topic = "/aux/kinect2/hd/image_color"
        # self.image_topic = "/aux/kinect2/qhd/image_color"
        self.depth_topic = "/aux/kinect2/sd/image_depth_rect"
        rospy.Subscriber(self.image_topic, Image, self.store_latest_color_image)
        rospy.Subscriber(self.depth_topic, Image, self.store_latest_depth_image)

        # Defining variables
        self.init_depth_image = None
        self.depth_image = None
        self.color_image = None

        # Needed to transform images
        self.bridge = CvBridge()

        # Saving initial images
        self.save_initial_color_image()
        self.save_initial_depth_image()

        # Starting multiprocessing
        def spin_thread():
            rospy.spin()

        self.cam_process = multiprocessing.Process(target=spin_thread)
        self.cam_process.start()

    def calc_object_loc(self):
        # Uses Andrews code to fit a cylinder and returns the center, height, radius and an image of the cylinder.
        ini_arr = np.array(self.init_depth_image)
        ini_depth = np.array(ini_arr[:, :, 0], 'float32')
        curr_arr = np.array(self.depth_image)
        curr_depth = np.array(curr_arr[:, :, 0], 'float32')
        return locate_cylinder.fit_cylinder(ini_depth, curr_depth)

    def save_initial_color_image(self):
        img = rospy.wait_for_message(self.image_topic, Image)
        self.color_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.color_image = np.flipud(self.color_image)
        self.color_image = np.fliplr(self.color_image)

    def save_initial_depth_image(self):
        img = rospy.wait_for_message(self.depth_topic, Image)
        self.depth_image = self.bridge.imgmsg_to_cv2(img, '16UC1')
        self.init_depth_image = self.depth_image

    def store_latest_color_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.color_image = cv_image

    def store_latest_depth_image(self, data):
        img = self.bridge.imgmsg_to_cv2(data, '16UC1')
        self.depth_image = img

    def get_color_image(self):
        return self.color_image

    def get_depth_image(self):
        return self.depth_image

    def end_process(self):
        self.cam_process.terminate()
        self.cam_process.join()
