#!/usr/bin/env python

# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

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
# import tensorflow_model_is_gripping.aolib.util as ut
# import tensorflow_model_is_gripping.grasp_net
# import tensorflow_model_is_gripping.grasp_params
# import tensorflow_model_is_gripping.aolib.img as ig

import grasp_cnn.aolib.util as ut
import grasp_cnn.grasp_net
import grasp_cnn.grasp_params
import grasp_cnn.params_v2
import grasp_cnn.aolib.img as ig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('run_experiment.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

__version__ = '0.0.1'
__author__ = 'Roberto Calandra <roberto.calandra@berkeley.edu>'

###############################
# Parameters experiment
################# ##############
COMPUTE_BG = False  # Store a new background image
bounds_table = np.array([[0.45, 0.65], [-0.25, 0.25]])  # X(min, max), Y(min, max) # TODO: this is too small!!!
grasping_force = [4, 25]  # min, max of the force [N] applied by the gripper when trying to grasp the object
xyz_bias = [+0.01, -0.02, 0]  # bias to compensate for sawyer-kinect calibration inaccuracies
# NAMEOBJECT = 'soft_blue_hexagon'
# NAMEOBJECT = 'jasmine_tea'
# NAMEOBJECT = '3d_printed_screw'
# NAMEOBJECT = 'yellow_berkeley_mug'
# NAMEOBJECT = 'wired_pen_container'
# NAMEOBJECT = 'glass_container_spices'
# NAMEOBJECT = 'staples_box'
NAMEOBJECT = 'logitech_mouse'
# NAMEOBJECT = 'mini_vase'
# NAMEOBJECT = 'coconut_tea_box'
# NAMEOBJECT = 'green_minecraft_toy'
# NAMEOBJECT = 'soft_plastic_brown_bear'
# NAMEOBJECT = 'green_asian_tea_cup'
THRESHOLD_GRASPING = 0.9

# Choose modality for prediction
image = False
image_VGG = True
if image_VGG:
    grasp_cnn.grasp_net.net_type = 'vgg'
gel_image = False
depth = False
gel_image_depth = False

###############################
# Parameters Gripper
###############################
# Gelsight adaptor v1
# lower_bound_z = 0.21  # When using the v1 of the weiss_gelsight_adaptor (the short one, with large grasp)
# height_gripper = 0.08  # v1 of the weiss_gelsight_adaptor
# Gelsight adaptor v2
lower_bound_z = 0.245  # When using the v2 of the weiss_gelsight_adaptor (the tall one, with smaller grasp)
height_gripper = 0.11  # v2 of the weiss_gelsight_adaptor


###############################

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
        xy_noise = 0.001
        shift = np.random.uniform(low=-xy_noise, high=xy_noise, size=3)
        shift_z_min = np.maximum(0.01, height_object - height_gripper)  # make sure that we don't hit with the gripper
        shift_z_max = height_object - 0.015  # small bias to avoid grasping air
        shift[2] = np.random.uniform(low=shift_z_min, high=shift_z_max)
        shift[2] = np.maximum(0, shift[2])  # Just for safety
        print('Z = [%f,%f] => %f' % (shift_z_min, shift_z_max, shift[2]))
        xyz = np.array([xy[0], xy[1], lower_bound_z]) + shift + xyz_bias
        orientation = orientation_downward(angle=np.random.uniform(0, np.pi))

    return xyz, orientation


def wait_for_key():
    rp = intera_interface.RobotParams()  # For logging
    rp.log_message("Press ESC to continue...")
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True


###############################



class grasper():
    def __init__(self, nameObject=''):

        self.first = True

        self.rp = intera_interface.RobotParams()  # For logging
        self.rp.log_message('')

        print('Make sure the correct object is printed below.')
        print('Object: %s' % nameObject)
        self.nameObject = nameObject
        # Make required initiations
        self.limb_name = "right"
        self.limb = None
        self.init_robot()
        self.gripper = self.init_gripper()

        # Requesting to start topics for KinectA
        self.rp.log_message('Launch topics for KinectA')
        self.rp.log_message('Please run the following command in a new terminal (in intera mode):')
        self.rp.log_message('rosrun kinect2_bridge kinect2_bridge')
        self.rp.log_message('')
        # Requesting to start topics for KinectB
        self.rp.log_message('Launch topics for KinectB')
        self.rp.log_message('Please run the following command in a new terminal (in intera mode) on the kinectbox02:')
        # rp.log_message('ssh k2')
        # rp.log_message('for pid in $(ps -ef | grep "kinect2_bridge" | awk "{print $2}"); do kill -9 $pid; done')
        self.rp.log_message('/home/rail/ros_ws/src/manu_kinect/start_KinectB.sh')
        self.rp.log_message('')
        # Start Topic for the Gelsight
        self.rp.log_message('Launch topic for GelsightA')
        self.rp.log_message('Please run the following command in a new terminal (in intera mode):')
        self.rp.log_message('roslaunch manu_sawyer gelsightA_driver.launch')
        self.rp.log_message('')
        self.rp.log_message('Launch topic for GelsightB')
        self.rp.log_message('Please run the following command in a new terminal (in intera mode):')
        self.rp.log_message('roslaunch manu_sawyer gelsightB_driver.launch')

        self.gelSightA = GelSightA()
        self.gelSightB = GelSightB()
        self.kinectA = KinectA(save_init=COMPUTE_BG)
        self.kinectB = KinectB()
        time.sleep(1)

        # Requests the user to place the object to be griped on the table.
        self.rp.log_message('Place the object to grasp on the table.')
        wait_for_key()

        self.start_experiment()  # Start grasping the object

    def init_robot(self):
        epilog = """
        See help inside the example with the '?' key for key bindings.
            """
        valid_limbs = self.rp.get_limb_names()
        if not valid_limbs:
            self.rp.log_message(("Cannot detect any limb parameters on this robot. "
                                 "Exiting."), "ERROR")
            return

        self.rp.log_message('Initializing node... ')
        rospy.init_node("move_and_grip")

        self.rp.log_message('Getting robot state...  ')
        self.rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = self.rs.state().enabled

        def clean_shutdown():
            print("\nExiting example.")
            if not init_state:
                self.rp.log_message('Disabling robot...')
                self.rs.disable()

        rospy.on_shutdown(clean_shutdown)

        rospy.loginfo("Enabling robot...")
        self.rs.enable()
        if not self.limb_name in valid_limbs:
            self.rp.log_message(("Right is not a valid limb on this robot. "
                                 "Exiting."), "ERROR")
            return
        limb = intera_interface.Limb(self.limb_name)
        limb.set_joint_position_speed(0.20)
        self.limb = limb
        # Move to a safe position
        self.goto_rest_pos()

    def init_gripper(self):
        # Requesting to start topics for gripper
        self.rp.log_message('Launch topics for gripper')
        self.rp.log_message('Please run the following command in a new terminal:')
        self.rp.log_message('roslaunch wsg_50_driver wsg_50_tcp_script.launch')
        return WSG50_manu.WSG50()

    def goto_randomized_grasping_location(self):
        # Randomize grasping location
        # move arm there:   self.goto_EE_xyz()
        pass

    def orientation_downward(self, angle):
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

    def goto_rest_pos(self, verbosity=1):
        """
        Move the arm to a safe rest position
        :param limb: link to the limb being used
        :param blocking: Bool. is it a blocking operation? (ie., do we wait until the end of the operation?)
        :param verbosity: verbosity level. >0 print stuff
        :return:
        """
        xyz_rest = [0.50, 0.50, 0.60]
        # if verbosity > 0:
        #     self.rp.log_message('Moving to rest position')
        self.goto_EE_xyz(xyz=xyz_rest, orientation=Orientations.DOWNWARD_ROTATED, verbosity=verbosity - 1,
                         rest_pos=True)

    def goto_EE_xyz(self, xyz, orientation=Orientations.DOWNWARD_ROTATED, verbosity=1, rest_pos=False):
        """
        Move the End-effector to the desired XYZ position and orientation, using inverse kinematic
        :param limb: link to the limb being used
        :param xyz: list or array [x,y,z] with the coordinates in XYZ positions in limb reference frame
        :param orientation:
        :param verbosity: verbosity level. >0 print stuff
        :return:
        """
        try:
            # if verbosity > 0:
            #     self.rp.log_message('Moving to x=%f y=%f z=%f' % (xyz[0], xyz[1], xyz[2]))
            if not rest_pos:
                # Make sure that the XYZ position is valid, and doesn't collide with the cage
                assert (xyz[0] >= bounds_table[0, 0]) and (xyz[0] <= bounds_table[0, 1]), 'X is outside of the bounds'
                assert (xyz[1] >= bounds_table[1, 0]) and (xyz[1] <= bounds_table[1, 1]), 'Y is outside of the bounds'
                assert (xyz[2] >= lower_bound_z), 'Z is outside of the bounds'
            des_pose = grip_and_record.inverse_kin.get_pose(xyz[0], xyz[1], xyz[2], orientation)
            curr_pos = self.limb.joint_angles()  # Measure current position
            joint_positions = grip_and_record.inverse_kin.get_joint_angles(des_pose, self.limb.name, curr_pos,
                                                                           use_advanced_options=True)  # gets joint positions
            self.limb.move_to_joint_positions(joint_positions)  # Send the command to the arm
        except UnboundLocalError:
            pose_dict = self.limb.endpoint_pose()
            pose_pos = pose_dict['position']
            current_xyz = [pose_pos.x, pose_pos.y, pose_pos.z]
            halfway_xyz = ((np.array(xyz) + np.array(current_xyz)) / 2.0).tolist()
            if np.linalg.norm(np.array(current_xyz) - np.array(halfway_xyz)) > 0.00001:
                time.sleep(0.2)
                if rest_pos:
                    self.goto_EE_xyz(halfway_xyz, orientation, rest_pos=True)
                    self.goto_EE_xyz(xyz, orientation, rest_pos=True)
                else:
                    self.goto_EE_xyz(halfway_xyz, orientation)
                    self.goto_EE_xyz(xyz, orientation)
            else:
                print("WoooOooOW")
                self.goto_EE_xyz([0.60, 0.0, 0.40], orientation, rest_pos=True)

    def predict_grasping_success(self, gel0_pre, gel1_pre, gel0_post, gel1_post, im0_pre, im0_post, depth0_pre,
                                 depth0_post):

        gpu = '/gpu:0'

        if self.first:
            if image:
                # net_pr = tensorflow_model_is_gripping.grasp_params.im_fulldata_v5()
                # checkpoint_file = '/home/manu/ros_ws/src/manu_research/manu_sawyer/src/tensorflow_model_is_gripping/training/net.tf-6499'

                self.net_pr = grasp_cnn.params_v2.im_v2()
                self.checkpoint_file = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/grasp_cnn/traning/ResNet_v1/im/net.tf-7000"

                self.net = grasp_cnn.grasp_net.NetClf(self.net_pr, self.checkpoint_file, gpu)

                self.first = False

            elif gel_image:
                # net_pr = tensorflow_model_is_gripping.grasp_params.gel_im_fulldata_v5()
                # checkpoint_file = '/home/manu/ros_ws/src/manu_research/manu_sawyer/src/tensorflow_model_is_gripping/training/full/net.tf-6499'

                self.net_pr = grasp_cnn.params_v2.gel_im_v2()
                self.checkpoint_file = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/grasp_cnn/traning/ResNet_v1/gel_im/net.tf-7000"

                self.net = grasp_cnn.grasp_net.NetClf(self.net_pr, self.checkpoint_file, gpu)

                self.first = False

            elif image_VGG:
                self.net_pr = grasp_cnn.params_v2.im_vgg_legacy_v2()
                self.checkpoint_file = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/grasp_cnn/traning/im-vgg-legacy-v2/training/net.tf-7000"

                self.net = grasp_cnn.grasp_net.NetClf(self.net_pr, self.checkpoint_file, gpu)

                self.first = False

            elif gel_image_depth:

                self.net_pr = grasp_cnn.params_v2.gel_im_depth_v2()
                self.checkpoint_file = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/grasp_cnn/traning/ResNet_v1/gel-im-depth/net.tf-7000"

                self.net = grasp_cnn.grasp_net.NetClf(self.net_pr, self.checkpoint_file, gpu)

                self.first = False

            elif depth:

                self.net_pr = grasp_cnn.params_v2.depth_v2()
                self.checkpoint_file = "/home/manu/ros_ws/src/manu_research/manu_sawyer/src/grasp_cnn/traning/ResNet_v1/depth/net.tf-7000"

                self.net = grasp_cnn.grasp_net.NetClf(self.net_pr, self.checkpoint_file, gpu)

                self.first = False

        # sc = lambda x: ig.scale(x, (224, 224))
        def sc(x):
            """ do a center crop (helps with gelsight) """
            x = ig.scale(x, (256, 256))
            return ut.crop_center(x, 224)

        crop = grasp_cnn.grasp_net.crop_kinect
        inputs = dict(
            gel0_pre=sc(gel0_pre),
            gel1_pre=sc(gel1_pre),
            gel0_post=sc(gel0_post),
            gel1_post=sc(gel1_post),
            im0_pre=sc(crop(im0_pre)),
            im0_post=sc(crop(im0_post)),
            depth0_pre=sc(crop(depth0_pre.astype('float32'))),
            depth0_post=sc(crop(depth0_post.astype('float32'))))

        prob = self.net.predict(**inputs)
        print("Probability: ", prob[1])
        return prob[1]

    def obj_func(self, x):
        """
        This is the function that evaluate the objective function, ie, the goodness of the grasping
        :param x: np.array of parameters to evaluate [EE_x,EE_y,EE_z,orientation,graspingforce]
        :return:
        """
        # Unpack parameters
        des_xyz = x[0:3]
        des_EE_xyz_above = des_xyz + np.array([0, 0, 0.2])
        des_orientation = self.orientation_downward(x[3])
        des_grasping_force = x[4]

        # Goto desired grasp position above
        # self.goto_EE_xyz(xyz=des_EE_xyz_above, orientation=des_orientation)

        # Get image from GelSight
        # GelSightA
        gelA_img_r_ini = self.gelSightA.get_image()
        gelA_img_r_ini = cv2.cvtColor(gelA_img_r_ini, cv2.COLOR_BGR2RGB)
        # GelSightB
        gelB_img_r_ini = self.gelSightB.get_image()
        gelB_img_r_ini = cv2.cvtColor(gelB_img_r_ini, cv2.COLOR_BGR2RGB)

        gel0_pre = gelA_img_r_ini
        gel1_pre = gelB_img_r_ini

        im0_pre = cv2.cvtColor(self.kinectA.get_color_image(), cv2.COLOR_BGR2RGB)
        depth0_pre = self.kinectA.get_depth_image()

        # Goto desired grasp position
        self.goto_EE_xyz(xyz=des_xyz, orientation=des_orientation)
        self.limb.set_joint_position_speed(0.10)

        # Attempt grasp
        self.grasp_object(force=des_grasping_force)

        time.sleep(1)

        gelA_img_r = self.gelSightA.get_image()
        gelA_img_r = cv2.cvtColor(gelA_img_r, cv2.COLOR_BGR2RGB)
        gelB_img_r = self.gelSightB.get_image()
        gelB_img_r = cv2.cvtColor(gelB_img_r, cv2.COLOR_BGR2RGB)
        gel0_post = gelA_img_r
        gel1_post = gelB_img_r

        im0_post = cv2.cvtColor(self.kinectA.get_color_image(), cv2.COLOR_BGR2RGB)
        depth0_post = self.kinectA.get_depth_image()

        # Predict goodness grasp
        out = self.predict_grasping_success(gel0_pre, gel1_pre, gel0_post, gel1_post, im0_pre, im0_post, depth0_pre,
                                            depth0_post)
        return out

    def reset_gripper(self, x):
        # des_xyz = x[0:3]
        # des_EE_xyz_above = des_xyz + np.array([0, 0, 0.2])
        # des_orientation = self.orientation_downward(x[3])

        self.gripper.open(speed=50)  # Open gripper
        time.sleep(0.5)
        # self.goto_EE_xyz(xyz=des_EE_xyz_above, orientation=des_orientation)

    def attempt_lift_(self, x):
        des_xyz = x[0:3]
        des_EE_xyz_above = des_xyz + np.array([0, 0, 0.2])
        des_orientation = self.orientation_downward(x[3])

        self.goto_EE_xyz(xyz=des_EE_xyz_above, orientation=des_orientation)

        time.sleep(4)

        self.goto_EE_xyz(xyz=des_xyz + np.array([0, 0, 0.04]), orientation=des_orientation)

        self.gripper.open(speed=100)
        time.sleep(0.5)

        self.goto_EE_xyz(xyz=des_EE_xyz_above, orientation=des_orientation)

    def grasp_object(self, force):
        """
        Close the gripper to grasp an object, up to the desired gasping force.
        :param gripper:
        :return:
        """
        print("Setting gripping force:", force)
        self.gripper.set_force(force)

        self.gripper.graspmove_nopending(width=5, speed=50)

        time.sleep(2)

    def start_experiment(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        # Fit a cylinder around the object with the Kinect and get location, etc.
        self.rp.log_message('Waiting for Kinect to stabilize')
        time.sleep(1)
        self.rp.log_message('Done')
        xyz_kinect, height_object, radius, obj_vis = self.kinectA.calc_object_loc()

        # Gettig image from KinectB
        top_img = self.kinectB.get_color_image()
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)

        # Get image from GelSight
        # GelSightA
        gelA_img_r_ini = self.gelSightA.get_image()
        gelA_img_r_ini = cv2.cvtColor(gelA_img_r_ini, cv2.COLOR_BGR2RGB)
        # GelSightB
        gelB_img_r_ini = self.gelSightB.get_image()
        gelB_img_r_ini = cv2.cvtColor(gelB_img_r_ini, cv2.COLOR_BGR2RGB)

        # Plot result from Kinect for visualisation of fitted cylinder
        # Plot pic from GelSight
        kinA_img = ax1.imshow(obj_vis)
        kinB_img = ax2.imshow(top_img)
        gelA_img = ax3.imshow(gelA_img_r_ini)
        gelB_img = ax4.imshow(gelB_img_r_ini)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        plt.draw()
        plt.ion()
        plt.show()

        # ----------------------------------------------------------------------------------

        # Transform from Kinect coordinates to Sawyer coordinates
        xyz_sawyer = transform(xyz_kinect[0], xyz_kinect[1], xyz_kinect[2]).reshape(3)

        #  Sample randomized gripper position based on the fitted cylinder data
        des_EE_xyz, des_orientation_EE = sample_from_cylinder(xyz_sawyer[0:2], height_object, radius)

        des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])
        self.goto_EE_xyz(xyz=des_EE_xyz_above, orientation=des_orientation_EE)
        self.limb.set_joint_position_speed(0.05)

        force = np.random.uniform(grasping_force[0], grasping_force[1])

        x = [des_EE_xyz[0], des_EE_xyz[1], des_EE_xyz[2], des_orientation_EE.y, force]  # Initial guess by the Kinect

        # ===========================================================================================================

        path = "/home/manu/ros_ws/src/manu_research/data/optimize_data/randomized_search_2/"
        nameFile = time.strftime("%Y-%m-%d_%H%M%S")
        file = open(path + nameFile + '.txt', 'w')
        file.write("Object name: " + NAMEOBJECT + "\n")
        file.write("gel_image: " + str(gel_image) + "\n")
        file.write("image: " + str(image) + "\n")
        file.write("image_VGG: " + str(image_VGG) + "\n")
        file.write("depth: " + str(depth) + "\n")
        file.write("gel_image_depth: " + str(gel_image_depth) + "\n")


        # # Randomize grasping location

        # Start optimization grasp
        # promising_attempt = False
        while True:
            # Predict grasping success
            grasping_success = self.obj_func(x)
            file.write(str(x) + " " + str(grasping_success) + "\n")
            # Decide what to do
            if grasping_success > THRESHOLD_GRASPING:
                self.attempt_lift_(x)
                break
            else:
                self.reset_gripper(x)

                noise = 0.003
                delta_xyz = np.random.uniform(low=-noise, high=noise, size=3)
                delta_z_min = np.maximum(0.01, height_object - height_gripper)
                delta_z_max = height_object - 0.015
                delta_xyz[2] = np.random.uniform(low=delta_z_min, high=delta_z_max)
                delta_xyz[2] = np.maximum(0, delta_xyz[2])  # Just for safety
                des_EE_xyz = np.array([des_EE_xyz[0], des_EE_xyz[1], lower_bound_z]) + delta_xyz
                angle = np.random.uniform(0, np.pi)
                force = np.random.uniform(grasping_force[0], grasping_force[1])
                x = [des_EE_xyz[0], des_EE_xyz[1], des_EE_xyz[2], angle, force]

        # ===========================================================================================================

        print("Was it a successful grasp? [y/n]")
        done = False
        while not done:
            c = grip_and_record.getch.getch()
            if c:
                if c in ['n']:
                    successful = False
                    done = True
                elif c in ['y']:
                    successful = True
                    done = True

        file.write("Successful attempt: " + str(successful))
        file.close()
        rospy.signal_shutdown("Example finished.")


if __name__ == '__main__':
    grasper(nameObject=NAMEOBJECT)
