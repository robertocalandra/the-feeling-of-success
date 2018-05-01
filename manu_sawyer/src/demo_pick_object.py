#!/usr/bin/env python

#inverse kinematics stuff
import grip_and_record.inverse_kin
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from grip_and_record.robot_utils import Orientations
import rospy
import intera_interface
from intera_interface import CHECK_VERSION
from intera_interface import (
    Lights,
    Cuff,
    RobotParams,
)
# import wsg_50_python
import numpy as np
import time
import grip_and_record.getch
import logging
# from WSG50 import WSG50
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler('demo_run_experiment.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


GRIPPER = 'WSG50'


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
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()
    if not limb_name in valid_limbs:
        rp.log_message(("Right is not a valid limb on this robot. "
                        "Exiting."), "ERROR")
        return
    limb = intera_interface.Limb(limb_name)
    # Move to a safe position
    goto_rest_pos(limb=limb, blocking=True)
    return limb


def init_gripper(gripper='Sawyer'):
    if gripper == 'Sawyer':
        pass
    if gripper == 'WSG50':
        gripper = WSG50()
    return gripper


def goto_rest_pos(limb, blocking=False):
    """
    Move the arm to a safe rest position
    :param limb:
    :param blocking: is it a blocking operation?
    :return:
    """
    goto_EE_xyz(limb=limb, xyz=[0.55, 0, 0.55], orientation=Orientations.DOWNWARD_ROTATED, blocking=blocking)


def goto_EE_xyz(limb, xyz, orientation=Orientations.DOWNWARD_ROTATED, blocking=False):
    """
    Move the End-effector to the desired XYZ position and orientation, using inverse kinematic
    :param limb: link to the limb being used
    :param xyz: list or array [x,y,z] with the coordinates in XYZ positions in limb reference frame
    :param orientation: 
    :return: 
    """
    # TODO: assert x,y,z
    des_pose = grip_and_record.inverse_kin.get_pose(xyz[0], xyz[1], xyz[2], orientation)
    curr_pos = limb.joint_angles()  # Measure current position
    joint_positions = grip_and_record.inverse_kin.get_joint_angles(des_pose, limb.name, curr_pos, use_advanced_options=False)  # gets joint positions
    limb.move_to_joint_positions(joint_positions)  # Send the command to the arm
    # TODO: implement blocking, in this moment we just wait to allow movemnt
    if blocking:
        time.sleep(2)


def main():
    limb_name = "right"
    limb = init_robot(limb_name=limb_name)
    gripper = init_gripper()
    # init_cuff(limb_name=limb_name)
    rp = intera_interface.RobotParams()  # For logging
    time.sleep(1)
    limb.set_joint_position_speed(0.12)  # Let' s move slowly...


    # Move arm to set position
    des_EE_xyz = np.array([0.55, 0, 0.3])
    des_orientation_EE = Orientations.DOWNWARD_ROTATED
    rp.log_message('Moving to x=%f y=%f' % (des_EE_xyz[0], des_EE_xyz[1]))
    goto_EE_xyz(limb=limb, xyz=des_EE_xyz, orientation=des_orientation_EE)


    #NOTE: MUST import: import getch
    print("Place object beetween the fingers and press Esc to close the gripper.")
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True


    gripper.close()  # TODO: make sure that the closure is properly done!
    time.sleep(1)  # give time to move

    # Raise the object
    goto_EE_xyz(limb=limb, xyz=des_EE_xyz + np.array([0, 0, 0.2]), orientation=des_orientation_EE)


    # Return object to the ground (a little bit higher)
    print("If the object is no longer grasped, clean the table and press Esc. Otherwise, Press Esc to return the object to the table.")
    done = False
    while not done and not rospy.is_shutdown():
        c = grip_and_record.getch.getch()
        if c:
            if c in ['\x1b', '\x03']:
                done = True

    # Return object to the ground (a little bit higher)
    # TODO: return the object to a new random position ?
    goto_EE_xyz(limb=limb, xyz=des_EE_xyz + np.array([0, 0, 0.02]), orientation=Orientations.DOWNWARD_ROTATED)
    time.sleep(0.5)
    gripper.open()  # Open gripper

    # data_recorder.end_processes()
    goto_rest_pos(limb=limb, blocking=True)  # Go to a safe place before shutting down

    rospy.signal_shutdown("Example finished.")


if __name__ == '__main__':
    main()
