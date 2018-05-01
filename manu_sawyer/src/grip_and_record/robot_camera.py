#!/usr/bin/env python

import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
import intera_interface
import thread
import multiprocessing
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import locate_cylinder


class RobotCam:
    def __init__(self, flipped=False):
        """
        Class for recording data from the Kinect2
        :param flipped:
        """
        self.flipped = flipped
        self.image_topic = "/kinect2/hd/image_color"
        self.depth_topic = "/kinect2/sd/image_depth_rect"
        rospy.Subscriber(self.image_topic, Image, self.store_latest_im)
        rospy.Subscriber(self.depth_topic, Image, self.store_latest_d_im)

        self.bridge = CvBridge()
        self.save_initial_image()

        self.init_depth = None
        self.save_initial_depth()

        def spin_thread():
            print "Started spin thread"
            rospy.spin()
        # thread.start_new(spin_thread, ())
        self.cam_process = multiprocessing.Process(target=spin_thread)
        self.cam_process.start()

    def calc_object_loc(self):
        ini_arr = np.array(self.init_depth)
        ini_depth = np.array(ini_arr[:, :, 0], 'float32')
        curr_arr = np.array(self.depth_image)
        curr_depth = np.array(curr_arr[:, :, 0], 'float32')
        return locate_cylinder.fit_cylinder(ini_depth, curr_depth)

    def save_initial_image(self):
        print('Saving initial Image')
        img = rospy.wait_for_message(self.image_topic, Image)
        self.latest_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        if self.flipped:
            self.latest_image = cv2.flip(self.latest_image , 0)

    def save_initial_depth(self):
        print('Saving initial Depth')
        img = rospy.wait_for_message(self.depth_topic, Image)
        self.depth_image = self.bridge.imgmsg_to_cv2(img, '16UC1')
        self.init_depth = self.depth_image

    def store_latest_im(self, data):
        # print ("CAMERA UPDATE")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self.flipped:
            self.latest_image = cv2.flip(cv_image, 0)
        else:
            self.latest_image = cv_image
        self.time_stamp_im = self._time_stamp()

    def store_latest_d_im(self, data):
        img = self.bridge.imgmsg_to_cv2(data, '16UC1')
        # img = img.astype(np.float32) / np.max(img) * 256
        # img = img.astype(np.uint8)
        # img = np.squeeze(img)
        self.depth_image = img

    def _time_stamp(self):
        return rospy.get_time()

    def get_latest_image(self):
        return self.latest_image

    def get_depth_image(self):
        return self.depth_image

    def end_process(self):
        self.cam_process.terminate()
        self.cam_process.join()
