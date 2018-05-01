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
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('run_experiment.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

bounds_table = np.array([[0.39, 0.85], [-0.27, 0.18]])  # X(min, max), Y(min, max)


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


def init_gripper(limb_name):
    try:
        gripper = intera_interface.Gripper(limb_name)
    except ValueError:
        rospy.logerr("Could not detect a gripper attached to the robot.")
        return
    if gripper.has_error():
        gripper.reboot()
    if not gripper.is_calibrated():
        gripper.calibrate()
    gripper.open()
    return gripper


def init_cuff(limb_name):
    class GripperConnect(object):
        """
        Connects wrist button presses to gripper open/close commands.
        Uses the Navigator callback feature to make callbacks to connected
        action functions when the button values change.
        """

        def __init__(self, arm, lights=True):
            """
            @type arm: str
            @param arm: arm of gripper to control
            @type lights: bool
            @param lights: if lights should activate on cuff grasp
            """
            self._arm = arm
            # inputs
            self._cuff = Cuff(limb=arm)
            # connect callback fns to signals
            self._lights = None
            if lights:
                self._lights = Lights()
                self._cuff.register_callback(self._light_action,
                                             '{0}_cuff'.format(arm))
            try:
                self._gripper = Gripper(arm)
                # if not (self._gripper.is_calibrated() or self._gripper.calibrate() == True):
                #     rospy.logerr("({0}_gripper) calibration failed.".format( self._gripper.name))
                #     raise
                self._cuff.register_callback(self._close_action, '{0}_button_upper'.format(arm))
                self._cuff.register_callback(self._open_action, '{0}_button_lower'.format(arm))
                rospy.loginfo("{0} Cuff Control initialized...".format(self._gripper.name))
            except:
                self._gripper = None
                msg = ("{0} Gripper is not connected to the robot."
                       " Running cuff-light connection only.").format(arm.capitalize())
                rospy.logwarn(msg)

        def _open_action(self, value):
            if value and self._gripper.is_ready():
                rospy.logdebug("gripper open triggered")
                self._gripper.open()
                if self._lights:
                    self._set_lights('red', False)
                    self._set_lights('green', True)

        def _close_action(self, value):
            if value and self._gripper.is_ready():
                rospy.logdebug("gripper close triggered")
                self._gripper.close()
                if self._lights:
                    self._set_lights('green', False)
                    self._set_lights('red', True)

        def _light_action(self, value):
            if value:
                rospy.logdebug("cuff grasp triggered")
            else:
                rospy.logdebug("cuff release triggered")
            if self._lights:
                self._set_lights('red', False)
                self._set_lights('green', False)
                self._set_lights('blue', value)

        def _set_lights(self, color, value):
            self._lights.set_light_state('head_{0}_light'.format(color), on=bool(value))
            self._lights.set_light_state('{0}_hand_{1}_light'.format(self._arm, color),
                                         on=bool(value))

    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
        return
    arms = (limb_name,) if limb_name != 'all_limbs' else valid_limbs[:-1]
    grip_ctrls = [GripperConnect(arm, True) for arm in arms]


def goto_rest_pos(limb, verbosity=1):
    """
    Move the arm to a safe rest position
    :param limb: link to the limb being used
    :param blocking: Bool. is it a blocking operation? (ie., do we wait until the end of the operation?)
    :param verbosity: verbosity level. >0 print stuff
    :return:
    """
    xyz_rest = [0.39, -0.27, 0.55]
    if verbosity > 0:
        rp = intera_interface.RobotParams()  # For logging
        rp.log_message('Moving to rest position')
    goto_EE_xyz(limb=limb, xyz=xyz_rest, orientation=Orientations.DOWNWARD_ROTATED, verbosity=verbosity - 1)


def goto_EE_xyz(limb, xyz, orientation=Orientations.DOWNWARD_ROTATED, verbosity=1):
    """
    Move the End-effector to the desired XYZ position and orientation, using inverse kinematic
    :param limb: link to the limb being used
    :param xyz: list or array [x,y,z] with the coordinates in XYZ positions in limb reference frame
    :param orientation:
    :param verbosity: verbosity level. >0 print stuff
    :return: 
    """
    # TODO: assert xyz numel == 3
    if verbosity > 0:
        rp = intera_interface.RobotParams()  # For logging
        rp.log_message('Moving to x=%f y=%f z=%f' % (xyz[0], xyz[1], xyz[2]))
    # Make sure that the XYZ position is valid, and doesn't collide with the cage
    assert (xyz[0] >= bounds_table[0, 0]) and (xyz[0] <= bounds_table[0, 1]), 'X is outside of the bounds'
    assert (xyz[1] >= bounds_table[1, 0]) and (xyz[1] <= bounds_table[1, 1]), 'Y is outside of the bounds'
    assert (xyz[2] >= 0.195), 'Z is outside of the bounds'
    des_pose = grip_and_record.inverse_kin.get_pose(xyz[0], xyz[1], xyz[2], orientation)
    curr_pos = limb.joint_angles()  # Measure current position
    joint_positions = grip_and_record.inverse_kin.get_joint_angles(des_pose, limb.name, curr_pos,
                                                                   use_advanced_options=False)  # gets joint positions
    limb.move_to_joint_positions(joint_positions)  # Send the command to the arm
    #  time.sleep(0.5)


def grasp_object(gripper):
    gripper.close()  # TODO: make sure that the closure is properly done!
    time.sleep(0.5)


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


# Fix this???
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
        xyz = np.array([des_xy[0], des_xy[1], 0.21])  # fix height

    if approach == 2:
        # Approach 2: directly sample angle and shift
        shift = np.random.uniform(low=-0.005, high=0.005, size=3)
        xyz = np.array([xy[0], xy[1], 0.195 + height_object / 4.]) + shift
        orientation = orientation_downward(angle=np.random.uniform(0, np.pi))

    return xyz, orientation


def test_quaternion():
    limb_name = "right"
    limb = init_robot(limb_name=limb_name)
    init_cuff(limb_name=limb_name)
    time.sleep(1)

    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(0), verbosity=1)
    time.sleep(1)
    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(np.pi / 8), verbosity=1)
    time.sleep(1)
    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(np.pi / 6), verbosity=1)
    time.sleep(1)
    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(np.pi / 4), verbosity=1)
    time.sleep(1)
    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(np.pi / 2), verbosity=1)
    time.sleep(1)
    goto_EE_xyz(limb, [0.5, 0, 0.55], orientation=orientation_downward(np.pi), verbosity=1)
    time.sleep(1)


def main():
    # Make required initiations
    limb_name = "right"
    limb = init_robot(limb_name=limb_name)
    limb.set_joint_position_speed(0.20)  # We can move fast
    gripper = init_gripper(limb_name=limb_name)
    init_cuff(limb_name=limb_name)
    rp = intera_interface.RobotParams()  # For logging
    time.sleep(0.5)

    ###############################
    # Pick the name of the object #
    ###############################
    name = "soft_red_cube"
    # name = "soft_elephant"
    # name = "qdasd"
    # name = "asd"
    # name = "Asdasde"
    print('Make sure the correct object is printed below.')
    print('Object: %s' % name)

    # Classifier for determining if gripper is gripping, using GelSight images.
    model_path = "/home/guser/catkin_ws/src/manu_research/manu_sawyer/src/press/training/net.tf-2600"
    net = press.NetClf(model_path, "/gpu:0")

    # Start topics for KinectA
    rp.log_message('Launching topics for KinectA')
    # For the KinectA
    os.system('killall -9 /home/guser/catkin_ws/devel/lib/kinect2_bridge/kinect2_bridge')
    os.system(
        'rosrun kinect2_bridge kinect2_bridge > /home/guser/catkin_ws/src/manu_research/temp/kinect_bridge_out.txt 2>&1 &')
    time.sleep(1)

    # Requesting to start topics for KinectB
    rp.log_message('Launch topics for KinectB')
    rp.log_message('Please run the following commands in a new terminal:')
    rp.log_message('ssh k1')
    #  rp.log_message('for pid in $(ps -ef | grep "kinect2_bridge" | awk "{print $2}"); do kill -9 $pid; done')
    rp.log_message(
        '~/catkin_ws/src/manu_sawyer/launch/startkinect_kinect1.sh > ~/catkin_ws/src/manu_sawyer/temp/kinect_bridge_out.txt 2>&1')
    rp.log_message('Press ESC when done.')
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True
    time.sleep(1)

    # Start Topic for the Gelsight
    rp.log_message('Launching topic for Gelsight...')
    os.system("for pid in $(ps -ef | grep 'gelsight' | awk '{print $2}'); do kill -9 $pid; done")
    os.system(
        'roslaunch manu_sawyer gelsightA_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightA_driver.txt 2>&1 &')
    os.system(
        'roslaunch manu_sawyer gelsightB_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightB_driver.txt 2>&1 &')
    time.sleep(10)
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
    kinectB = KinectB()
    time.sleep(0.5)

    # Start data recorder
    data_recorder = DataRecorder(limb=limb, gripper=gripper, GelSightA=gelSightA, GelSightB=gelSightB, KinectA=kinectA,
                                 KinectB=kinectB)
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

    while run:
        Iter += 1

        # Fit a cylinder around the object with the Kinect and get location, etc.
        rp.log_message('Waiting for Kinect to stabilize')
        time.sleep(0.5)
        print("Done")
        xyz_kinect, height_object, radius, obj_vis = kinectA.calc_object_loc()


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
        else:
            kinA_img.set_data(obj_vis)
            kinB_img.set_data(top_img)
            gelA_img.set_data(gelA_img_r_ini)
            gelB_img.set_data(gelB_img_r_ini)
        plt.draw()
        plt.ion()
        plt.show()

        ####################################################
        # Give an appropriate name to the files we save to #
        ####################################################

        # Initialize recording to file
        nameFile = time.strftime("%Y-%m-%d_%H%M%S")
        thread = threading.Thread(target=data_recorder.init_record, args=(nameFile,))
        thread.start()

        # Save the name of the object
        data_recorder.set_object_name(name)

        # Transform from Kinect coordinates to Sawyer coordinates
        xyz_sawyer = transform(xyz_kinect[0], xyz_kinect[1], xyz_kinect[2]).reshape(3)

        # Temporary fix due to some calibration issues
        xyz_sawyer[1] -= 0.02

        # If randomize_gripper_position is True, we grip the object with some randomness
        if randomize_gripper_position:
            #  Sample randomized gripper position based on the fitted cylinder data
            des_EE_xyz, des_orientation_EE = sample_from_cylinder(xyz_sawyer[0:2], height_object, radius)
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])
        else:
            des_orientation_EE = Orientations.DOWNWARD_ROTATED
            des_EE_xyz = xyz_sawyer
            des_EE_xyz[2] = 0.195 + height_object / 4.
            des_EE_xyz_above = des_EE_xyz + np.array([0, 0, 0.2])

        # Move above the object
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=Orientations.DOWNWARD_ROTATED)

        # Rotate the gripper
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=des_orientation_EE)

        # Move down to the object and record the location of the EE
        limb.set_joint_position_speed(0.10)  # Let' s move slowly...
        goto_EE_xyz(limb=limb, xyz=des_EE_xyz, orientation=des_orientation_EE)
        data_recorder.set_location_of_EE_when_attempting_gripping(des_EE_xyz)

        # Grip the object and record the time
        grasp_object(gripper)
        time_of_gripping_attempt = rospy.get_time()
        data_recorder.set_time_of_gripping_attempt(time_of_gripping_attempt)

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

        # Record the time
        time_of_gripper_in_air = rospy.get_time()
        data_recorder.set_time_of_gripper_in_air(time_of_gripper_in_air)

        # Check whether the object still is in the gripper
        gelA_img_r = gelSightA.get_image()
        is_gripping_gripper = gripper.is_gripping()
        is_gripping_GelSightA = False
        if net.predict(gelA_img_r, gelA_img_r_ini) > 0.1:
            is_gripping_GelSightA = True
        is_gripping = is_gripping_gripper or is_gripping_GelSightA

        rp.log_message('Am I gripping? %s' % is_gripping)

        # Record the result
        data_recorder.set_is_gripping(is_gripping)

        if is_gripping:
            # If we are still gripping the object we return object to the ground at a random location

            # Compute random x and y coordinates
            r_x = np.random.uniform(0.1, 0.9, 1)
            r_y = np.random.uniform(0.1, 0.9, 1)
            x_min = bounds_table[0, 0]
            x_max = bounds_table[0, 1]
            y_min = bounds_table[1, 0]
            y_max = bounds_table[1, 1]
            x_r = r_x * x_min + (1 - r_x) * x_max
            y_r = r_y * y_min + (1 - r_y) * y_max

            # Calculate appropriate z coordinate
            height_grasp = 0.195 + height_object / 4.

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

            # Go up, but a little higher than before
            des_EE_xyz_above[2] = 0.55
            goto_EE_xyz(limb=limb, xyz=des_EE_xyz_above, orientation=random_orientation)

            # Go back to rest position
            goto_rest_pos(limb=limb)
        else:
            # If we are not gripping the object, i.e. the grasp failed, we move to the resting position immediately.
            gripper.open()
            goto_rest_pos(limb=limb)

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

    data_recorder.end_processes()  # Stop recorder

    os.system('killall -9 /home/guser/catkin_ws/devel/lib/kinect2_bridge/kinect2_bridge')  # Closes KinectA topics
    # kinectA.end_process()
    # kinectB.end_process()
    # gelSightB.end_process()
    # gelSightA.end_process()
    rospy.signal_shutdown("Example finished.")


def testGelSights():
    # rospy.init_node('Testing')
    init_robot('right')
    # Start Topic for the Gelsight
    os.system("for pid in $(ps -ef | grep 'gelsight' | awk '{print $2}'); do kill -9 $pid; done")
    os.system(
        'roslaunch manu_sawyer gelsightA_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightA_driver.txt 2>&1 &')
    os.system(
        'roslaunch manu_sawyer gelsightB_driver.launch > /home/guser/catkin_ws/src/manu_research/temp/gelsightB_driver.txt 2>&1 &')
    time.sleep(10)
    gelSightA = GelSightA()
    time.sleep(10)
    gelSightB = GelSightB()
    time.sleep(20)
    gelA_ini = gelSightA.get_image()
    gelB_ini = gelSightB.get_image()

    model_path = "/home/guser/catkin_ws/src/manu_research/manu_sawyer/src/press/training/net.tf-2600"

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
