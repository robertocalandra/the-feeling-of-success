#!/usr/bin/env python
from robot_camera import RobotCam
import rospy
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np

rospy.init_node('Testing')
kinect = RobotCam(True)
print "ini done"

rate = rospy.Rate(30)
# counter = 0
try:
    while (True):
        # Capture frame-by-frame
        init_time = time.time()
        frame = kinect.get_latest_image()
        # point = kinect.get_point_cloud()
        # print(point)
        #
        print ("TIME ELAPSED", time.time() - init_time)
        #
        # Our operations on the frame come here
        # imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # depth_img = depth_img.astype(np.float32) / np.max(depth_img) * 512
        # depth_img = depth_img.astype(np.uint8)
        # depth_img = np.squeeze(depth_img)
        #
        # # Display the resulting frame
        cv2.imshow('frame', frame)
        #
        #
        # # plt.imshow(frame)
        # # plt.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # counter += 1
        rate.sleep()
except KeyboardInterrupt:
    pass
kinect.end_process()
