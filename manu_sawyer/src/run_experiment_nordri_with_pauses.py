#!/usr/bin/env python

import grip_and_record.inverse_kin
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from data_recorder import DataRecorder
import data_recorder as dr

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
import WSG50_manu
import tensorflow_model_is_gripping.press as press
import pylab
import cv2
import time
import random
import multiprocessing
import tensorflow_model_is_gripping.aolib.util as ut

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('run_experiment.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

__version__ = '1.0.1'

# Parameters experiment
###############################
COMPUTE_BG = False  # Store a new background image
save_parallel = False
bounds_table = np.array([[0.45, 0.65], [-0.25, 0.25]])  # X(min, max), Y(min, max) # TODO: this is too small!!!
grasping_force = [4, 25]  # min, max of the force [N] applied by the gripper when trying to grasp the object
time_waiting_in_air = 4  # Number of seconds the object is held in the air to determine if the grasp is stable.
xyz_bias = [0.005, -0.01, 0]  # bias to compensate for kinect-sawyer calibration inaccuracies

FIX_Y = 0.04
FIX_ANGLE = 1
###############################
# Parameters Gripper
###############################
# Gelsight adaptor v1
# lower_bound_z = 0.21  # When using the v1 of the weiss_gelsight_adaptor (the short one, with large grasp)
# height_gripper = 0.08  # v1 of the weiss_gelsight_adaptor
# Gelsight adaptor v2
lower_bound_z = 0.242  # When using the v2 of the weiss_gelsight_adaptor (the tall one, with smaller grasp)
height_gripper = 0.11  # v2 of the weiss_gelsight_adaptor

###############################
# Pick the name of the object #
###############################
# name = 'peptobismol'
# name = 'soda_can'
# name = 'purple_meausure_1_cup'
# name = 'green_plastic_cup'
# name = "soft_red_cube"
# name = "soft_elephant"
# name = "soft_zebra"
# name = "blue_cup"
# name = "wooden_pyramid"
# name = "french_dip"
# name = "red_bull"
# name = "metal_can"
# name = "spam"
# name = "soft_blue_cylinder"
# name = "wooden_cube"
# name = "soda_can"
# name = "rubics_cube"
# name = "plastic_duck"
# name = "glass_candle_holder"
# name = "black_metallic_candle_cage"
# name = "aspirin"
# name = "ponds_dry_skin_cream"
# name = "edge_shave_gel"
# name = "ogx_shampoo"
# name = "isopropyl_alcohol"
# name = "baby_cup"
# name = "kong_dog_toy"
# name = "dark_blue_sphere"
# name = "bandaid_box"
# name = "angry_bird"
# name = "hand_soap" # cylinder-fitting fails
# name = "plastic_whale"
# name = "plastic_cow"
# name = "monster_truck"
# name = "plastic_mushroom"
# name = "mesh_container" #-> basket?
# name = "bag_pack" #-> forslutas (?)
# name = "chocolate_shake"
# name = "brown_paper_cup"
# name = "brown_paper_cup_2_upside" # Two stacked
# name = "toy_person_with_hat" # bad, too small
# name = "webcam_box"
# name = "playdoh_container"
# name = "pig"
# name = "stuffed_beachball"
# name = "tuna_can"
# name = "bottom_of_black_metallic_candle_cage" # fails
# name = "metal_cylinder_with_holes"
# name = "set_small_plastic_men_yellow_construction_worker"
# name = "wiry_sphere"
# name = "translucent_turquoise_cup" # cylinder-fitting overestimates size (maybe due to translucency)
# name = "green_and_black_sphere"
# name = "blue_translucent_glass_cup" #cylinder-fitting totally failed here
# name = "plastic_sheep"
# name = 'feathered_ball'
# name = "plastic_chicken"
# name = 'blueish_plastic_cup'
# name = "set_small_plastic_men_police_man"
# name = 'orange_plastic_castle' # -> from the toy box in the cabinet
# name = 'pink_glass_glass' # the one painted by roberto
# name = 'blue_painted_glass'
# name = 'soft_blue_hexagon'
# name = "egg_crate_foam"
# name = "dog_toy_ice_cream_cone"
# name = "onion"
# name = "axe_body_spray"
# name = "candle_in_glass"
# name = "tomato_paste_in_metal_can"
# name = "small_coffe_cup"
name = "yellow_wooden_robot"
# name = "international_travel_adapter"
# name = "lemon"
# name = "muffin"
# name = "lime"
# name = "potato"
# name = "red_apple"


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
    limb.set_joint_position_speed(0.03)
    # Move to a safe position
    goto_rest_pos(limb=limb)

    return limb


def wait_for_key():
    rp = intera_interface.RobotParams()  # For logging
    rp.log_message("Press ESC to continue...")
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True


def init_gripper():
    rp = intera_interface.RobotParams()  # For logging

    # Requesting to start topics for gripper
    rp.log_message('Launch topics for gripper')
    rp.log_message('Please run the following command in a new terminal:')
    rp.log_message('roslaunch wsg_50_driver wsg_50_tcp_script.launch')
    # rp.log_message('Press ESC when done.')
    # done = False
    # while not done and not rospy.is_shutdown():
    #     c = grip_and_record.getch.getch()
    #     if c:
    #         if c in ['\x1b', '\x03']:
    #             done = True
    time.sleep(1)
    return WSG50_manu.WSG50()


def goto_rest_pos(limb, verbosity=1):
    """
    Move the arm to a safe rest position
    :param limb: link to the limb being used
    :param blocking: Bool. is it a blocking operation? (ie., do we wait until the end of the operation?)
    :param verbosity: verbosity level. >0 print stuff
    :return:
    """
    xyz_rest = [0.50, 0.50, 0.60]
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
            assert (xyz[2] >= lower_bound_z), 'Z is outside of the bounds'
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
        if np.linalg.norm(np.array(current_xyz) - np.array(halfway_xyz)) > 0.00001:
            time.sleep(0.2)
            if rest_pos:
                goto_EE_xyz(limb, halfway_xyz, orientation, rest_pos=True)
                goto_EE_xyz(limb, xyz, orientation, rest_pos=True)
            else:
                goto_EE_xyz(limb, halfway_xyz, orientation)
                goto_EE_xyz(limb, xyz, orientation)
        else:
            print("WoooOooOW")
            goto_EE_xyz(limb, [0.60, 0.0, 0.40], orientation, rest_pos=True)


def grasp_object(gripper, data_recorder):
    """
    Close the gripper to grasp an object, up to the desired gasping force.
    :param gripper:
    :return:
    """
    force = random.randint(grasping_force[0], grasping_force[1])
    data_recorder.set_set_gripping_force(force)
    print("Setting gripping force:", force)
    gripper.set_force(force)

    # Reset the gripper
    gripper.homing()

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
        angle_gripper = np.pi / 2 + (np.pi - (angles[1] - angles[0]) / 2) + angles[0]  # TODO: compute angle gripper y = ax + b
        # rp.log_message('Moving to x=%f y=%f z=%f' % (des_xy[0], des[1], xyz[2]))
        angle_gripper = 0
        orientation = orientation_downward(angle=angle_gripper)
        xyz = np.array([des_xy[0], des_xy[1], 0.25])  # fix height

    if approach == 2:
        # Approach 2: directly sample angle and shift
        xy_noise = 0.001
        shift = np.random.uniform(low=-xy_noise, high=xy_noise, size=3)
        shift_z_min = np.maximum(0.01, height_object-height_gripper)  # make sure that we don't hit with the gripper
        shift_z_max = height_object-0.015  # small bias to avoid grasping air
        shift[2] = FIX_Y #np.random.uniform(low=shift_z_min, high=shift_z_max)
        shift[2] = np.maximum(0, shift[2])  # Just for safety
        print('Z = [%f,%f] => %f' %(shift_z_min, shift_z_max, shift[2]))
        xyz = np.array([xy[0], xy[1], lower_bound_z]) + shift + xyz_bias
        orientation = orientation_downward(angle=FIX_ANGLE)

    return xyz, orientation


def main():

    print('Make sure the correct object is printed below.')
    print('Object: %s' % name)

    # Make required initiations
    limb_name = "right"
    limb = init_robot(limb_name=limb_name)
    gripper = init_gripper()

    rp = intera_interface.RobotParams()  # For logging
    rp.log_message('')

    # Classifier for determining if gripper is gripping, using GelSight images.
    model_path = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/tensorflow_model_is_gripping/training/net.tf-4600"  # net.tf-2600
    net = press.NetClf(model_path, "/gpu:0")

    # Requesting to start topics for KinectA
    rp.log_message('Launch topics for KinectA')
    rp.log_message('Please run the following command in a new terminal (in intera mode):')
    rp.log_message('rosrun kinect2_bridge kinect2_bridge')
    rp.log_message('')
    # Requesting to start topics for KinectB
    rp.log_message('Launch topics for KinectB')
    rp.log_message('Please run the following command in a new terminal (in intera mode) on the kinectbox02:')
    # rp.log_message('ssh k2')
    # rp.log_message('for pid in $(ps -ef | grep "kinect2_bridge" | awk "{print $2}"); do kill -9 $pid; done')
    rp.log_message('/home/rail/ros_ws/src/manu_kinect/start_KinectB.sh')
    rp.log_message('')
    # Start Topic for the Gelsight
    rp.log_message('Launch topic for GelsightA')
    rp.log_message('Please run the following command in a new terminal (in intera mode):')
    rp.log_message('roslaunch manu_sawyer gelsightA_driver.launch')
    rp.log_message('')
    rp.log_message('Launch topic for GelsightB')
    rp.log_message('Please run the following command in a new terminal (in intera mode):')
    rp.log_message('roslaunch manu_sawyer gelsightB_driver.launch')
    rp.log_message('')
    # wait_for_key()

    gelSightA = GelSightA()
    gelSightB = GelSightB()

    kinectA = KinectA(save_init=COMPUTE_BG)
    kinectB = KinectB()
    time.sleep(1)

    from multiprocessing.pool import ThreadPool
    pool = multiprocessing.pool.ThreadPool(processes=1)

    # Requests the user to place the object to be griped on the table.
    rp.log_message('Place the object to grasp on the table.')
    wait_for_key()

    # Setup for main loop #

    # For plotting purposes
    Iter = 0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # The name of the variable says the most
    ask_user_to_continue_or_end_after_each_iteration = False

    # Whether to randomize gripper position and orientation or not when gripping the object
    randomize_gripper_position = True

    # Condition variable for the loop
    run = True

    #################
    # The main loop #
    #################

    comp_task = None
    while run:
        Iter += 1
        start_time = rospy.get_time()

        # Fit a cylinder around the object with the Kinect and get location, etc.
        rp.log_message('Waiting for Kinect to stabilize')
        # time.sleep(0.5)
        rp.log_message('Done')
        xyz_kinect, height_object, radius, obj_vis = kinectA.calc_object_loc()

        data_recorder = DataRecorder(limb=limb, gripper=gripper, GelSightA=gelSightA, GelSightB=gelSightB,
                                     KinectA=kinectA,
                                     KinectB=kinectB)

        # Initialize recording to file
        nameFile = time.strftime("%Y-%m-%d_%H%M%S")
        thread = threading.Thread(target=data_recorder.init_record, args=(nameFile,))
        thread.start()
        time.sleep(0.2)

        # Record cylinder data
        data_recorder.set_cylinder_data(xyz_kinect, height_object, radius)

        # Save the name of the object
        data_recorder.set_object_name(name)

        # Gettig image from KinectB
        top_img = kinectB.get_color_image()
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)

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
            kinB_img = ax2.imshow(top_img)
            gelA_img = ax3.imshow(gelA_img_r_ini)
            gelB_img = ax4.imshow(gelB_img_r_ini)
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
        else:
            kinA_img.set_data(obj_vis)
            kinB_img.set_data(top_img)
            gelA_img.set_data(gelA_img_r_ini)
            gelB_img.set_data(gelB_img_r_ini)
        plt.draw()
        plt.ion()
        plt.show()

        # Transform from Kinect coordinates to Sawyer coordinates
        xyz_sawyer = transform(xyz_kinect[0], xyz_kinect[1], xyz_kinect[2]).reshape(3)

        # If randomize_gripper_position is True, we grip the object with some randomness
        if randomize_gripper_position:
            #  Sample randomized gripper position based on the fitted cylinder data
            des_EE_xyz, des_orientation_EE = sample_from_cylinder(xyz_sawyer[0:2], height_object, radius)
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.18])
        else:
            des_orientation_EE = Orientations.DOWNWARD_ROTATED
            des_EE_xyz = xyz_sawyer
            des_EE_xyz[2] = lower_bound_z + height_object / 2
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.18])

        # Move above the object
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=Orientations.DOWNWARD_ROTATED)
        # Rotate the gripper
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        wait_for_key()

        # Record the time pre grasping
        time_pre_grasping = rospy.get_time()
        data_recorder.set_time_pre_grasping(time_pre_grasping)

        # Move down to the object and record the location of the EE
        limb.set_joint_position_speed(0.03)  # Let' s move slowly...
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz, orientation=des_orientation_EE)
        data_recorder.set_location_of_EE_at_grasping(des_EE_xyz)
        data_recorder.set_angle_of_EE_at_grasping(des_orientation_EE.y)
        wait_for_key()

        # Grasp the object and record the time
        grasp_object(gripper, data_recorder)
        time.sleep(0.5)  # This is crucial!!!! keep it!
        time_at_grasping = rospy.get_time()
        wait_for_key()
        data_recorder.set_time_at_grasping(time_at_grasping)

        # Get image from GelSights and update plot
        gelA_img_r = gelSightA.get_image()
        gelA_img_r = cv2.cvtColor(gelA_img_r, cv2.COLOR_BGR2RGB)
        gelB_img_r = gelSightB.get_image()
        gelB_img_r = cv2.cvtColor(gelB_img_r, cv2.COLOR_BGR2RGB)
        gelA_img.set_data(gelA_img_r)
        gelB_img.set_data(gelB_img_r)
        plt.draw()
        plt.ion()
        plt.show()

        # Raise the object slightly above current position
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        time.sleep(0.5)
        wait_for_key()

        # Get image from GelSights and update plot
        gelA_img_r = gelSightA.get_image()
        gelA_img_r = cv2.cvtColor(gelA_img_r, cv2.COLOR_BGR2RGB)
        gelB_img_r = gelSightB.get_image()
        gelB_img_r = cv2.cvtColor(gelB_img_r, cv2.COLOR_BGR2RGB)
        gelA_img.set_data(gelA_img_r)
        gelB_img.set_data(gelB_img_r)
        plt.draw()
        plt.ion()
        plt.show()

        # Record the time
        time_post1_grasping = rospy.get_time()
        data_recorder.set_time_post1_grasping(time_post1_grasping)

        # Wait a little
        time.sleep(time_waiting_in_air)

        # Check whether the object still is in the gripper
        gelA_img_r = gelSightA.get_image()
        gelB_img_r = gelSightB.get_image()
        pred_A = net.predict(gelA_img_r, gelA_img_r_ini)
        pred_B = net.predict(gelB_img_r, gelB_img_r_ini)
        data_recorder.set_probability_A(pred_A)
        data_recorder.set_probability_B(pred_B)
        print("Pred A:", pred_A)
        print("Pred B:", pred_B)

        gelA_img_r = cv2.cvtColor(gelA_img_r, cv2.COLOR_BGR2RGB)
        gelB_img_r = cv2.cvtColor(gelB_img_r, cv2.COLOR_BGR2RGB)
        gelA_img.set_data(gelA_img_r)
        gelB_img.set_data(gelB_img_r)
        plt.draw()
        plt.ion()
        plt.show()

        is_gripping_A = False
        if pred_A >= 0.9:
            is_gripping_A = True

        # is_gripping_B = False
        # if pred_B >= 0.9:
        #     is_gripping_B = True

        is_gripping_gripper = False
        gripper_force = gripper.get_force()
        if gripper_force >= 2:
            is_gripping_gripper = True
        print("Getting gripping force:", gripper_force)

        # is_gripping = is_gripping_A or is_gripping_B or is_gripping_gripper
        is_gripping = is_gripping_A or is_gripping_gripper

        rp.log_message('Am I gripping? %s' % is_gripping)

        # Record the result
        data_recorder.set_is_gripping(is_gripping)

        # Record the time
        time_post2_grasping = rospy.get_time()
        data_recorder.set_time_post2_grasping(time_post2_grasping)

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

            # (comment below to go back to the original position)
            des_EE_xyz = np.array((x_r, y_r, des_EE_xyz[2]))

            # Move above the new random position
            des_EE_xyz_above = des_EE_xyz.copy()
            des_EE_xyz_above[2] = des_EE_xyz[2] + 0.2
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)

            # Randomize the rotation too
            random_orientation = orientation_downward(np.random.uniform(0, np.pi))
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=random_orientation)

            # Go down to the random position and let go of the object
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz + np.array([0, 0, 0.02]), orientation=random_orientation)
            gripper.open(speed=100)  # Open gripper
            limb.set_joint_position_speed(0.03)  # We can move faster

            # Go up, but a little higher than before
            des_EE_xyz_above[2] = 0.60
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=random_orientation)
            limb.set_joint_position_speed(0.03)  # We can move faster

            # Get image from GelSights and update plot
            gelA_img_r = gelSightA.get_image()
            gelA_img_r = cv2.cvtColor(gelA_img_r, cv2.COLOR_BGR2RGB)
            gelB_img_r = gelSightB.get_image()
            gelB_img_r = cv2.cvtColor(gelB_img_r, cv2.COLOR_BGR2RGB)
            gelA_img.set_data(gelA_img_r)
            gelB_img.set_data(gelB_img_r)
            plt.draw()
            plt.ion()
            plt.show()

            # Go back to rest position
            goto_rest_pos(limb=limb)
        else:
            # If we are not gripping the object, i.e. the grasp failed, we move to the resting position immediately.
            gripper.open(speed=200)
            limb.set_joint_position_speed(0.03)  # We can move faster
            goto_rest_pos(limb=limb)

        # Reset the gripper
        gripper.homing()

        if ask_user_to_continue_or_end_after_each_iteration:
            print("Do you wish to run another grasp? [y/n]")
            done = False
            while not done and not rospy.is_shutdown():
                c = grip_and_record.getch.getch()
                if c:
                    if c in ['n', '\x1b', '\x03']:
                        done = True
                        run = False
                    elif c in ['y']:
                        done = True

        # Stop recording data for this iteration
        data_recorder.stop_record()
        thread.join()

        if save_parallel:
            if comp_task is not None:
                ut.tic('Waiting for comp_task')
                if not comp_task.wait():
                    raise RuntimeError('Compression task failed!')
                ut.toc()
            comp_task = dr.CompressionTask(data_recorder, pool)
            comp_task.run_async()
        else:
            comp_task = dr.CompressionTask(data_recorder, pool)
            comp_task.run_sync()

        import gc; gc.collect()

        end_time = rospy.get_time()
        print("Time of grasp:", end_time - start_time)

    # Stop recorder
    # TODO: move end_processes outside of data_recorder
    data_recorder.end_processes()

    # kinectA.end_process()
    # kinectB.end_process()
    # gelSightB.end_process()
    # gelSightA.end_process()
    rospy.signal_shutdown("Example finished.")


def testGelSights():
    # rospy.init_node('Testing')
    init_robot('right')
    # Start Topic for the Gelsight
    # os.system("for pid in $(ps -ef | grep 'gelsight' | awk '{print $2}'); do kill -9 $pid; done")
    # os.system(
    #     'roslaunch manu_sawyer gelsightA_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightA_driver.txt 2>&1 &')
    # os.system(
    #     'roslaunch manu_sawyer gelsightB_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightB_driver.txt 2>&1 &')
    # time.sleep(10)
    gelSightA = GelSightA()
    # time.sleep(10)
    gelSightB = GelSightB()
    time.sleep(10)
    gelA_ini = gelSightA.get_image()
    gelB_ini = gelSightB.get_image()

    model_path = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/tensorflow_model_is_gripping/training/net.tf-3000"

    net = press.NetClf(model_path, "/gpu:0")

    cmap = pylab.cm.RdYlGn
    i = 0
    while True:
        frameA = gelSightA.get_image()
        frameB = gelSightB.get_image()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        probA = net.predict(frameA, gelA_ini)
        probB = net.predict(frameB, gelB_ini)
        colorA = map(int, 255 * np.array(cmap(probA))[:3])
        colorB = map(int, 255 * np.array(cmap(probB))[:3])

        cv2.putText(frameA, '%.2f' % probA, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorA)
        cv2.putText(frameB, '%.2f' % probB, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colorB)
        cv2.imshow('frameA', frameA)
        cv2.imshow('frameB', frameB)


if __name__ == '__main__':
    # test_quaternion()
    # testGelSights()
    main()
