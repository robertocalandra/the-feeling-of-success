#!/usr/bin/env python


import grip_and_record.inverse_kin
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from data_recorder import DataRecorder
from grip_and_record.robot_utils import Orientations

import rospy
import intera_interface
from intera_interface import CHECK_VERSION
from intera_interface import (
    Gripper,
    Lights,
    Cuff,
    RobotParams,
)
import numpy as np
from transform import transform
import time
import grip_and_record.getch
import grip_and_record.locate_cylinder
import os
import matplotlib.pyplot as plt
from KinectA import KinectA
from KinectB import KinectB
import logging
import threading
from GelSightA import GelSightA
from GelSightB import GelSightB
import tensorflow_model_is_gripping.press as press
import pylab
import cv2
import h5py
import WSG50_manu
import random

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('collect_data_for_Andrews_net.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

bounds_table = np.array([[0.35, 0.65], [-0.25, 0.25]])  # X(min, max), Y(min, max)


def init_robot(limb_name):
    epilog = """
    See help inside the example with the '?' key for key bindings.
        """
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
        return

    rp.log_message('Initializing node... ')
    rospy.init_node("move_and_grip")

    rp.log_message('Getting robot state...  ')
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example.")
        if not init_state:
            rp.log_message('Disabling robot...')
            rs.disable()

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()
    if not limb_name in valid_limbs:
        rp.log_message(("Right is not a valid limb on this robot. "
                        "Exiting."), "ERROR")
        return
    limb = intera_interface.Limb(limb_name)
    limb.set_joint_position_speed(0.2)
    # Move to a safe position
    goto_rest_pos(limb=limb)

    return limb


def init_gripper():
    rp = intera_interface.RobotParams()  # For logging

    # Requesting to start topics for gripper
    os.system("for pid in $(ps -ef | grep 'wsg_50_tcp_script' | awk '{print $2}'); do kill -9 $pid; done")
    rp.log_message('Launch topics for gripper')
    rp.log_message('Please run the following command in a new terminal:')
    rp.log_message(
        'roslaunch wsg_50_driver wsg_50_tcp_script.launch')
    rp.log_message('Press ESC when done.')
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True
    time.sleep(1)
    print("done")
    return WSG50_manu.WSG50()


def goto_rest_pos(limb, verbosity=1):
    """
    Move the arm to a safe rest position
    :param limb: link to the limb being used
    :param blocking: Bool. is it a blocking operation? (ie., do we wait until the end of the operation?)
    :param verbosity: verbosity level. >0 print stuff
    :return:
    """
    xyz_rest = [0.60, 0.00, 0.60]
    if verbosity > 0:
        rp = intera_interface.RobotParams()  # For logging
        rp.log_message('Moving to rest position')
    goto_EE_xyz(limb=limb, xyz=xyz_rest, orientation=Orientations.DOWNWARD_ROTATED, verbosity=verbosity - 1,
                rest_pos=True)


def goto_EE_xyz(limb, xyz, orientation=Orientations.DOWNWARD_ROTATED, verbosity=1, rest_pos=False):
    """
    Move the End-effector to the desired XYZ position and orientation, using inverse kinematic
    :param limb: link to the limb being used
    :param xyz: list or array [x,y,z] with the coordinates in XYZ positions in limb reference frame
    :param orientation:
    :param verbosity: verbosity level. >0 print stuff
    :return:
    """
    try:
        if verbosity > 0:
            rp = intera_interface.RobotParams()  # For logging
            rp.log_message('Moving to x=%f y=%f z=%f' % (xyz[0], xyz[1], xyz[2]))
        if not rest_pos:
            # Make sure that the XYZ position is valid, and doesn't collide with the cage
            assert (xyz[0] >= bounds_table[0, 0]) and (xyz[0] <= bounds_table[0, 1]), 'X is outside of the bounds'
            assert (xyz[1] >= bounds_table[1, 0]) and (xyz[1] <= bounds_table[1, 1]), 'Y is outside of the bounds'
            assert (xyz[2] >= 0.215), 'Z is outside of the bounds'
        des_pose = grip_and_record.inverse_kin.get_pose(xyz[0], xyz[1], xyz[2], orientation)
        curr_pos = limb.joint_angles()  # Measure current position
        joint_positions = grip_and_record.inverse_kin.get_joint_angles(des_pose, limb.name, curr_pos,
                                                                       use_advanced_options=True)  # gets joint positions
        limb.move_to_joint_positions(joint_positions)  # Send the command to the arm
    except UnboundLocalError:
        pose_dict = limb.endpoint_pose()
        pose_pos = pose_dict['position']
        current_xyz = [pose_pos.x, pose_pos.y, pose_pos.z]
        halfway_xyz = ((np.array(xyz) + np.array(current_xyz)) / 2.0).tolist()
        if np.linalg.norm(np.array(current_xyz) - np.array(halfway_xyz)) > 0.0005:
            if rest_pos:
                goto_EE_xyz(limb, halfway_xyz, orientation, rest_pos=True)
                goto_EE_xyz(limb, xyz, orientation, rest_pos=True)
            else:
                goto_EE_xyz(limb, halfway_xyz, orientation)
                goto_EE_xyz(limb, xyz, orientation)
        else:
            print("WoooOooOW")
            goto_EE_xyz(limb, [0.50, 0.60, 0.30], orientation, rest_pos=True)


def grasp_object(gripper):

    # Reset the gripper
    gripper.homing()

    force = random.randint(5, 50)
    print("Setting gripping force:", force)
    gripper.set_force(force)
    gripper.grasp()


def orientation_downward(angle):
    """
    Return the quaternion for the gripper orientation
    :param angle: [rad]
    :return:
    """
    angle = np.remainder(angle, np.pi)  # Remap any angle to [0, +pi]
    orientation = Quaternion(
        x=1,
        y=angle,
        z=0,
        w=0,
    )
    return orientation

def sample_from_cylinder(xy, height_object=0.25, radius=0.1):
    """
    Randomly sample a grasping position from a cylinder
    :param xy: x,y coordinates of the base/center of the cylinder
    :param height_object: height of the cylinder
    :param radius: radius of the cylinder
    :return:
    """
    approach = 2
    xy = np.array(xy)
    # TODO: assert things are the right dimension

    if approach == 1:
        # Approach 1: sample two points from the circumference, and the grasp is the line connecting them
        angles = np.random.uniform(0, 2 * np.pi, 2)  # sample 2 points in terms of angles [rad]
        xy_points = xy + [radius * np.sin(angles), radius * np.cos(angles)]  # convert them to xy position
        # compute line between points and corresponding EE position
        des_xy = np.sum(xy_points, 0) / 2  # Middle point
        angle_gripper = np.pi / 2 + (np.pi - (angles[1] - angles[0]) / 2) + angles[
            0]  # TODO: compute angle gripper y = ax + b
        # rp.log_message('Moving to x=%f y=%f z=%f' % (des_xy[0], des[1], xyz[2]))
        angle_gripper = 0
        orientation = orientation_downward(angle=angle_gripper)
        xyz = np.array([des_xy[0], des_xy[1], 0.25])  # fix height

    if approach == 2:
        # Approach 2: directly sample angle and shift
        shift = np.random.uniform(low=-0.02, high=0.02, size=3)
        shift[2] = np.random.uniform(low=-0.005, high=0.005)
        xyz = np.array([xy[0], xy[1], 0.21 + np.random.uniform(0.01, height_object/2.0)]) + shift
        orientation = orientation_downward(angle=np.random.uniform(0, np.pi))

    return xyz, orientation

def main():
    # Make required initiations
    limb_name = "right"
    limb = init_robot(limb_name=limb_name)
    gripper = init_gripper()
    rp = intera_interface.RobotParams()  # For logging

    # Start topics for KinectA
    rp.log_message('Launching topics for KinectA')
    # For the KinectA
    os.system("for pid in $(ps -ef | grep 'kinect2_bridge' | awk '{print $2}'); do kill -9 $pid; done")
    os.system(
        'rosrun kinect2_bridge kinect2_bridge > /home/manu/ros_ws/src/manu_research/temp/kinect_bridge_out.txt 2>&1 &')
    print("done")

    # Requesting to start topics for GelSights
    rp.log_message('Launch topics for GelSights. Skip if the topic are already running.')
    rp.log_message('Please run the following commands in a new terminal (first run intera.sh):')
    rp.log_message('roslaunch manu_sawyer gelsightA_driver.launch')
    rp.log_message('Please run the following commands in a new terminal(first run intera.sh):')
    rp.log_message('roslaunch manu_sawyer gelsightB_driver.launch')
    rp.log_message('Press ESC when done.')
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True
    time.sleep(2)
    gelSightA = GelSightA()
    gelSightB = GelSightB()

    # Requests the user to clear the table of all objects.
    # Takes initial picture with KinectA.
    # The initial picture is needed to localize the object on the table later.
    rp.log_message('Please remove all objects from the table and then press ESC.')
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True
    kinectA = KinectA()
    time.sleep(0.5)

    # Requests the user to place the object to be griped on the table.
    rp.log_message('Place the object to grasp on the table and press ESC.')
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True

    # Setup for main loop #

    # The directory where the data is going to get saved
    path = "/home/manu/ros_ws/src/manu_research/data/for_Andrew/"

    # For plotting purposes
    Iter = 0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Whether to randomize gripper position and orientation or not when gripping the object
    randomize_gripper_position = True

    # Condition variable for the loop
    run = True

    #################
    # The main loop #
    #################

    while run:
        Iter += 1

        # Fit a cylinder around the object with the Kinect and get location, etc.
        rp.log_message('Waiting for Kinect to stabilize')
        time.sleep(3)
        print("Done")
        xyz_kinect, height_object, radius, obj_vis = kinectA.calc_object_loc()

        # Get image from GelSight
        # GelSightA
        gelA_img_r_ini = gelSightA.get_image()
        gelA_img_r_ini = cv2.cvtColor(gelA_img_r_ini, cv2.COLOR_BGR2RGB)
        # GelSightB
        gelB_img_r_ini = gelSightB.get_image()
        gelB_img_r_ini = cv2.cvtColor(gelB_img_r_ini, cv2.COLOR_BGR2RGB)

        # Plot result from Kinect for visualisation of fitted cylinder
        # Plot pic from GelSight
        if Iter == 1:
            kinA_img = ax1.imshow(obj_vis)
            gelA_img = ax3.imshow(gelA_img_r_ini)
            gelB_img = ax4.imshow(gelB_img_r_ini)
        else:
            kinA_img.set_data(obj_vis)
            gelA_img.set_data(gelA_img_r_ini)
            gelB_img.set_data(gelB_img_r_ini)
        plt.draw()
        plt.ion()
        plt.show()

        # Initialize recording to file
        nameFile = time.strftime("%Y-%m-%d_%H%M%S")
        file = h5py.File(path + nameFile + ".hdf5", "w")

        # Transform from Kinect coordinates to Sawyer coordinates
        xyz_sawyer = transform(xyz_kinect[0], xyz_kinect[1], xyz_kinect[2]).reshape(3)

        # If randomize_gripper_position is True, we grip the object with some randomness
        if randomize_gripper_position:
            #  Sample randomized gripper position based on the fitted cylinder data
            des_EE_xyz, des_orientation_EE = sample_from_cylinder(xyz_sawyer[0:2], height_object, radius)
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])
        else:
            des_orientation_EE = Orientations.DOWNWARD_ROTATED
            des_EE_xyz = xyz_sawyer
            des_EE_xyz[2] = 0.21 + np.random.uniform(0.01, height_object/2)
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])

        # Move above the object
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=Orientations.DOWNWARD_ROTATED)

        # Get image from GelSight
        # GelSightA
        gelA_img_r_pre = gelSightA.get_image()
        gelA_img_r_pre = cv2.cvtColor(gelA_img_r_pre, cv2.COLOR_BGR2RGB)
        # GelSightB
        gelB_img_r_pre = gelSightB.get_image()
        gelB_img_r_pre = cv2.cvtColor(gelB_img_r_pre, cv2.COLOR_BGR2RGB)

        # Save the images
        file.create_dataset("GelSightA_image_pre_gripping", data=gelA_img_r_pre)
        file.create_dataset("GelSightB_image_pre_gripping", data=gelB_img_r_pre)

        # Plot pics from GelSight
        gelA_img.set_data(gelA_img_r_pre)
        gelB_img.set_data(gelB_img_r_pre)
        plt.draw()
        plt.ion()
        plt.show()

        # Rotate the gripper
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)

        # Move down to the object and record the location of the EE
        limb.set_joint_position_speed(0.15)  # Let' s move slowly...
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz, orientation=des_orientation_EE)

        # Grip the object
        grasp_object(gripper)

        # Raise the object slightly above current position
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        time.sleep(0.5)

        # Get image from GelSights and update plot
        gelA_img_r_post = gelSightA.get_image()
        gelA_img_r_post = cv2.cvtColor(gelA_img_r_post, cv2.COLOR_BGR2RGB)
        gelB_img_r_post = gelSightB.get_image()
        gelB_img_r_post = cv2.cvtColor(gelB_img_r_post, cv2.COLOR_BGR2RGB)
        gelA_img.set_data(gelA_img_r_post)
        gelB_img.set_data(gelB_img_r_post)
        plt.draw()
        plt.ion()
        plt.show()

        # Save the images
        file.create_dataset("GelSightA_image_post_gripping", data=gelA_img_r_post)
        file.create_dataset("GelSightB_image_post_gripping", data=gelB_img_r_post)

        # Print gripper force
        gripper_force = gripper.get_force()
        print("Getting gripping force:", gripper_force)

        print("Is the robot holding the object in its gripper? [y/n]")
        done = False
        while not done and not rospy.is_shutdown():
            c = grip_and_record.getch.getch()
            if c:
                if c in ['n']:
                    is_gripping = False
                    done = True
                elif c in ['y']:
                    is_gripping = True
                    done = True

        print(is_gripping)
        file.create_dataset("is_gripping", data=np.asarray([is_gripping]))

        if is_gripping:
            # If we are still gripping the object we return object to the ground at a random location

            # Compute random x and y coordinates
            r_x = np.random.uniform(0.15, 0.85, 1)
            r_y = np.random.uniform(0.15, 0.85, 1)
            x_min = bounds_table[0, 0]
            x_max = bounds_table[0, 1]
            y_min = bounds_table[1, 0]
            y_max = bounds_table[1, 1]
            x_r = r_x * x_min + (1 - r_x) * x_max
            y_r = r_y * y_min + (1 - r_y) * y_max

            # Calculate appropriate z coordinate
            height_grasp = 0.21 + np.random.uniform(0.01, height_object/2)

            # (comment below to go back to the original position)
            des_EE_xyz = np.array((x_r, y_r, height_grasp))

            # Move above the new random position
            des_EE_xyz_above = des_EE_xyz.copy()
            des_EE_xyz_above[2] = des_EE_xyz[2] + 0.2
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)

            # Randomize the rotation too
            random_orientation = orientation_downward(np.random.uniform(0, np.pi))
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=random_orientation)

            # Go down to the random position and let go of the object
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz + np.array([0, 0, 0.02]), orientation=random_orientation)
            gripper.open()  # Open gripper
            time.sleep(0.5)
            limb.set_joint_position_speed(0.20)

            # Go up, but a little higher than before
            des_EE_xyz_above[2] = 0.40
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=random_orientation)

            # Go back to rest position
            goto_rest_pos(limb=limb)
        else:
            # If we are not gripping the object, i.e. the grasp failed, we move to the resting position immediately.
            gripper.open()
            limb.set_joint_position_speed(0.20)
            goto_rest_pos(limb=limb)
        #
        # Iter += 1
        #
        # # Fit a cylinder around the object with the Kinect and get location, etc.
        # rp.log_message('Waiting for Kinect to stabilize')
        # time.sleep(0.5)
        # print("Done")
        # xyz_kinect, height_object, radius, obj_vis = kinectA.calc_object_loc()
        #
        # # Get image from GelSight
        # # GelSightA
        # gelA_img_r_ini = gelSightA.get_image()
        # gelA_img_r_ini = cv2.cvtColor(gelA_img_r_ini, cv2.COLOR_BGR2RGB)
        # # GelSightB
        # gelB_img_r_ini = gelSightB.get_image()
        # gelB_img_r_ini = cv2.cvtColor(gelB_img_r_ini, cv2.COLOR_BGR2RGB)
        #
        # # Plot result from Kinect for visualisation of fitted cylinder
        # # Plot pic from GelSight
        # if Iter == 1:
        #     kinA_img = ax1.imshow(obj_vis)
        #     gelA_img = ax3.imshow(gelA_img_r_ini)
        #     gelB_img = ax4.imshow(gelB_img_r_ini)
        # else:
        #     kinA_img.set_data(obj_vis)
        #     gelA_img.set_data(gelA_img_r_ini)
        #     gelB_img.set_data(gelB_img_r_ini)
        # plt.draw()
        # plt.ion()
        # plt.show()
        #
        #
        # # Initialize recording to file
        # nameFile = time.strftime("%Y-%m-%d_%H%M%S")
        # file = h5py.File(path + nameFile + ".hdf5", "w")
        #
        # # Transform from Kinect coordinates to Sawyer coordinates
        # xyz_sawyer = transform(xyz_kinect[0], xyz_kinect[1], xyz_kinect[2]).reshape(3)
        #
        # # If randomize_gripper_position is True, we grip the object with some randomness
        # if randomize_gripper_position:
        #     #  Sample randomized gripper position based on the fitted cylinder data
        #     des_EE_xyz, des_orientation_EE = sample_from_cylinder(xyz_sawyer[0:2], height_object, radius)
        #     des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])
        # else:
        #     des_orientation_EE = Orientations.DOWNWARD_ROTATED
        #     des_EE_xyz = xyz_sawyer
        #     des_EE_xyz[2] = 0.21 + np.random.uniform(0.01, height_object/2)
        #     des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])
        #
        # # Move above the object
        # goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=Orientations.DOWNWARD_ROTATED)
        #
        # # Get image from GelSight
        # # GelSightA
        # gelA_img_r_pre = gelSightA.get_image()
        # gelA_img_r_pre = cv2.cvtColor(gelA_img_r_pre, cv2.COLOR_BGR2RGB)
        # # GelSightB
        # gelB_img_r_pre = gelSightB.get_image()
        # gelB_img_r_pre = cv2.cvtColor(gelB_img_r_pre, cv2.COLOR_BGR2RGB)
        #
        # # Save the images
        # file.create_dataset("GelSightA_image_pre_gripping", data=gelA_img_r_pre)
        # file.create_dataset("GelSightB_image_pre_gripping", data=gelB_img_r_pre)
        #
        # # Plot pics from GelSight
        # gelA_img.set_data(gelA_img_r_pre)
        # gelB_img.set_data(gelB_img_r_pre)
        # plt.draw()
        # plt.ion()
        # plt.show()
        #
        # # Rotate the gripper
        # goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        #
        # # Move down to the object and record the location of the EE
        # limb.set_joint_position_speed(0.15)  # Let' s move slowly...
        # goto_EE_xyz(limb=limb, xyz=des_EE_xyz, orientation=des_orientation_EE)
        #
        # # Grip the object and then open the gripper
        # grasp_object(gripper)
        # gripper.open()
        # limb.set_joint_position_speed(0.20)
        #
        # # Raise the object slightly above current position
        # goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        # time.sleep(0.5)
        #
        # # Get image from GelSights and update plot
        # gelA_img_r_post = gelSightA.get_image()
        # gelA_img_r_post = cv2.cvtColor(gelA_img_r_post, cv2.COLOR_BGR2RGB)
        # gelB_img_r_post = gelSightB.get_image()
        # gelB_img_r_post = cv2.cvtColor(gelB_img_r_post, cv2.COLOR_BGR2RGB)
        # gelA_img.set_data(gelA_img_r_post)
        # gelB_img.set_data(gelB_img_r_post)
        # plt.draw()
        # plt.ion()
        # plt.show()
        #
        # # Save the images
        # file.create_dataset("GelSightA_image_post_gripping", data=gelA_img_r_post)
        # file.create_dataset("GelSightB_image_post_gripping", data=gelB_img_r_post)
        #
        # print("Is the robot holding the object in its gripper?")
        # is_gripping = False
        # print(is_gripping)
        #
        # file.create_dataset("Is gripping?", data=np.asarray([is_gripping]))
        #
        # # We are not gripping the object, i.e. the grasp failed, we move to the resting position immediately.
        # goto_rest_pos(limb=limb)

    os.system('killall -9 /home/guser/catkin_ws/devel/lib/kinect2_bridge/kinect2_bridge')  # Closes KinectA topics
    kinectA.end_process()
    gelSightA.end_process()
    gelSightB.end_process()
    rospy.signal_shutdown("Example finished.")


if __name__ == '__main__':
    main()
