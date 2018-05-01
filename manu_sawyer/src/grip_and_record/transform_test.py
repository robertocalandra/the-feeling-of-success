#!/usr/bin/env python
from robot_camera import RobotCam
import rospy
import time
import cv2
import numpy as np
import os


# TODO: run kinect script if not already running
# ps aux | grep kinect
# os.system('killall -9 /home/rcalandra/ros_ws/devel/lib/kinect2_bridge/kinect2_bridge')
# os.system('rosrun kinect2_bridge kinect2_bridge > /home/rcalandra/ros_ws/src/manu_sawyer/temp/kinect_bridge_out.txt 2>&1 &')
# time.sleep(10)

# TODO: Ask to clean table, and take first image...
# TODO: watch out for gripper

rospy.init_node('Testing')
print('Init camera')
kinect = RobotCam()
print("ini done")

rate = rospy.Rate(30)


try:
    # Capture frame-by-frame
    i = 0
    while True:
        init_time = time.time()
        print('Getting kinect data')
        frame = kinect.get_depth_image()
        # camera_depth = np.asarray(self.robot_cam.get_depth_image())
        # depth = dset['camera_depth'][:, :, 0]
        # depths.append(np.array(depth, 'float32'))

        #depth_img = frame.astype(np.float32) / np.max(frame) * 512
        #depth_img = depth_img.astype(np.uint8)
        #depth_img = np.squeeze(depth_img)
        # # Display the resulting frame
        if i == 0:
            time.sleep(1)
        #
        # # plt.imshow(frame)
        # # plt.show()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            # os.system('killall -9 /home/rcalandra/ros_ws/devel/lib/kinect2_bridge/kinect2_bridge')
            break

        center, height, d, obj_vis = kinect.calc_object_loc()
        print('center =', center)
        print('radius =', d)
        print('height =', height)
        cv2.imshow('frame', obj_vis)
        rate.sleep()
        i += 1
except KeyboardInterrupt:
    print('Error')
    pass
kinect.end_process()


def on_scan(self, data):
    '''
    @data: Input pointcloud data. A list of tuple (x,y,X,Y,Z)
    Callback every time point cloud data is received.
    '''
    print("=== Picking points ===")
    # side setup
    X_MIN = 0
    X_MAX = 600
    Y_MIN = 50
    Y_MAX = 300

    # x_min = 0.32
    # x_max = 0.85
    # y_min = -0.80
    # y_max = -0.30
    # z_min = Z_LIMIT
    # z_max = 0.05
    x_min = 0.32
    x_max = 0.85
    y_min = -0.82
    y_max = -0.27
    z_min = Z_LIMIT
    z_max = 0.05

    valid = []

    scan_range = product(xrange(X_MIN, X_MAX), xrange(Y_MIN, Y_MAX))

    valid_points = 0
    p = pc2.read_points(data, skip_nans=False, uvs=scan_range)

    scan_range = product(xrange(X_MIN, X_MAX), xrange(Y_MIN, Y_MAX))

    valid_points_img = self.cur_img.copy()

    XY = []
    Z = []

    for X, Y in scan_range:
        point = p.next()
        x, y, z, d = np.array(point)
        if all([x < x_max, x > x_min, y > y_min, y < y_max]):
            valid.append((x, y, z, X, Y))
            valid_points_img[:, Y, X] = np.array([0, 0, 0])
            if all([x < x_max - 0.05, x > x_min + 0.05]):  # extra constraints
                XY.append((1.0, x, y))
                Z.append(z)

    self.valid_points_publisher.send(valid_points_img)

    A = np.array(XY)
    b = np.array(Z)
    print A.shape, b.shape
    coeffs, _, _, _ = np.linalg.lstsq(A, b)

    print "saving", list(coeffs)
    np.save("plane_coeffs", coeffs)

    self.depth_subscriber.unregister()

    print("Done.")
