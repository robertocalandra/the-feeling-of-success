#!/usr/bin/env python

# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import argparse
import importlib

import rospy
from dynamic_reconfigure.server import Server
from std_msgs.msg import Empty

import intera_interface
from intera_interface import CHECK_VERSION

import numpy as np
from dotmap import DotMap
# from .PD import PID
import h5py

from geometry_msgs.msg import Quaternion
from grip_and_record.inverse_kin import get_joint_angles, get_pose


class JointSprings(object):
    """
    Virtual Joint Springs class for torque example.
    @param limb: limb on which to run joint springs example
    @param reconfig_server: dynamic reconfigure server
    JointSprings class contains methods for the joint torque example allowing
    moving the limb to a neutral location, entering torque mode, and attaching
    virtual springs.
    """

    def __init__(self, reconfig_server, limb="right"):
        self._dyn = reconfig_server

        # control parameters
        self._freq = 20.0  # Hz
        self._missed_cmds = 5.0  # Missed cycles before triggering timeout

        # create our limb instance
        self._limb = intera_interface.Limb(limb)

        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._start_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + limb + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")
        # self.controller = PID(dX=1, dU=1, P=1)

        # self.EE_target = [0.6, 0, 0.60]
        # quaternion = Quaternion(x=0.707, y=0.707, z=0, w=0)
        # computed_angles = get_joint_angles(pose=get_pose(x=self.EE_target[0], y=self.EE_target[1], z=self.EE_target[2],
        #                                                  o=quaternion),
        #                                    use_advanced_options=False)
        # self.target = [
        #     computed_angles['right_j6'],
        #     computed_angles['right_j5'],
        #     computed_angles['right_j4'],
        #     computed_angles['right_j3'],
        #     computed_angles['right_j2'],
        #     computed_angles['right_j1'],
        #     computed_angles['right_j0'],
        # ]

        self.target = self.joint_dict_to_arr(self._limb.joint_angles())

        self.P = np.array([25, 25, 30, 35, 45, 65, 45])  # P gains
        self.K = np.array([0.5, 0.5, 2, 3, 3, 5, 5])

    def joint_dict_to_arr(self, joint_dict):
        return np.array([
            joint_dict['right_j6'],
            joint_dict['right_j5'],
            joint_dict['right_j4'],
            joint_dict['right_j3'],
            joint_dict['right_j2'],
            joint_dict['right_j1'],
            joint_dict['right_j0'],
        ])

    def set_torque(self, torques):
        torques = np.array(torques)
        torques[np.isinf(torques)] = 0
        torques[np.isnan(torques)] = 0
        bounds = np.array([10, 10, 10, 10, 10, 10, 10])
        torques = np.maximum(-bounds, np.minimum(bounds, torques))  # Bounds torques
        cmd = dict()
        for i, joint in enumerate(self._start_angles.keys()):
            # spring portion
            cmd[joint] = torques[i]
        self._limb.set_joint_torques(cmd)  # Send command to the robot

    def _update_parameters(self):
        for joint in self._limb.joint_names():
            self._springs[joint] = self._dyn.config[joint[-2:] + '_spring_stiffness']
            self._damping[joint] = self._dyn.config[joint[-2:] + '_damping_coefficient']

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """
        # get latest spring constants
        self._update_parameters()

        # disable cuff interaction
        self._pub_cuff_disable.publish()

        # create our command dict
        cmd = np.zeros(7)
        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()

        # calculate current forces
        for i, joint in enumerate(self._start_angles.keys()):
            # # spring portion
            # cmd[i] = self._springs[joint] * (self._start_angles[joint] -
            #                                        cur_pos[joint])
            # # damping portion
            # cmd[i] -= self._damping[joint] * cur_vel[joint]

            e = self.target[i] - cur_pos[joint]
            cmd[i] = self.P[i] * e
            cmd[i] = cmd[i] - self.K[i] * cur_vel[joint]
        # print('Torques applied: ' + cmd)
        # print(cur_pos)

        state = np.array([
            cur_pos['right_j6'],
            cur_pos['right_j5'],
            cur_pos['right_j4'],
            cur_pos['right_j3'],
            cur_pos['right_j2'],
            cur_pos['right_j1'],
            cur_pos['right_j0'],
        ])
        self.set_torque(cmd)  # command robot

    def move_to_neutral(self):
        """
        Moves the limb to neutral location.
        """
        self._limb.move_to_neutral()

    def attach_springs(self):
        """
        Switches to joint torque mode and attached joint springs to current
        joint positions.
        """
        # record initial joint angles
        self._start_angles = self._limb.joint_angles()

        # set control rate
        control_rate = rospy.Rate(self._freq)

        # for safety purposes, set the control rate command timeout.
        # if the specified number of command cycles are missed, the robot
        # will timeout and return to Position Control Mode
        self._limb.set_command_timeout((1.0 / self._freq) * self._missed_cmds)

        # loop at specified rate commanding new joint torques
        while not rospy.is_shutdown():
            if not self._rs.state().enabled:
                # rospy.logerr("Joint torque example failed to meet "
                #              "specified control rate timeout.")
                break


            self._update_forces()
            control_rate.sleep()

    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        # Save logs to file
        # self.log_file.create_dataset("action_dataset", data=np.array(self.logs.action))
        # self.log_file.create_dataset("state_dataset", data=np.array(self.logs.state))
        # self.log_file.attrs['frequency'] = self._freq
        # self.log_file.close()
        # print('\nLogs saved to file: %s' % self.log_nameFile)

        print("Exiting example...")
        self._limb.exit_control_mode()


def main():
    """RSDK Joint Torque Example: Joint Springs
    Moves the default limb to a neutral location and enters
    torque control mode, attaching virtual springs (Hooke's Law)
    to each joint maintaining the start position.
    Run this example and interact by grabbing, pushing, and rotating
    each joint to feel the torques applied that represent the
    virtual springs attached. You can adjust the spring
    constant and damping coefficient for each joint using
    dynamic_reconfigure.
    """
    # Querying the parameter server to determine Robot model and limb name(s)
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
    robot_name = intera_interface.RobotParams().get_robot_name().lower().capitalize()
    # Parsing Input Arguments
    arg_fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument(
        "-l", "--limb", dest="limb", default=valid_limbs[0],
        choices=valid_limbs,
        help='limb on which to attach joint springs'
    )
    args = parser.parse_args(rospy.myargv()[1:])
    # Grabbing Robot-specific parameters for Dynamic Reconfigure
    config_name = ''.join([robot_name, "JointSpringsExampleConfig"])
    config_module = "intera_examples.cfg"
    cfg = importlib.import_module('.'.join([config_module, config_name]))
    # Starting node connection to ROS
    print("Initializing node... ")
    rospy.init_node("sdk_joint_torque_springs_{0}".format(args.limb))
    dynamic_cfg_srv = Server(cfg, lambda config, level: config)

    js = JointSprings(dynamic_cfg_srv, limb=args.limb)  # <- MOVE!
    # register shutdown callback
    rospy.on_shutdown(js.clean_shutdown)
    # js.move_to_neutral()
    js.attach_springs()



if __name__ == "__main__":
    main()
