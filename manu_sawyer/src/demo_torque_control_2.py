from __future__ import division, print_function, absolute_import

from time import strftime, localtime

import argparse
import importlib

import rospy
from dynamic_reconfigure.server import Server
from std_msgs.msg import Empty

import intera_interface
from intera_interface import CHECK_VERSION

import numpy as np
from dotmap import DotMap
import h5py

from geometry_msgs.msg import Quaternion
from inverse_kin import get_joint_angles, get_pose


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
    cfg = importlib.import_module('.'.join([config_module,config_name]))
    # Starting node connection to ROS
    print("Initializing node... ")
    rospy.init_node("sdk_joint_torque_springs_{0}".format(args.limb))
    dynamic_cfg_srv = Server(cfg, lambda config, level: config)

    dc = DataCollector(dynamic_cfg_srv, limb=args.limb)  # <- MOVE!
    # register shutdown callback
    rospy.on_shutdown(dc.clean_shutdown)
    for i in range(30):
        print("Starting a new trajectory.")
        dc.collect_data(n_iters=10)
        dc.reset_to_initial_pos()


class DataCollector(object):
    """
    Collects dynamics model data by moving the end-effector to random positions in front
    of the robot.
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
        self.initial_angles = joint_dict_to_arr(self._limb.joint_angles())

        self.P = np.array([5, 5, 10, 15, 30, 20, 25])  # P gains
        self.K = np.array([0.7, 0.2, 1.63, 0.5, 3.5, 4.7, 6.0])

        self.num_completed_trajectories = 0

        self.log_nameFile = strftime("/home/kchua/ros_ws/src/roberto_sawyer/data/sawyer_data/%Y-%m-%d|%H:%M:%S.hdf5", localtime())
        self.log_file = h5py.File(self.log_nameFile, "w")
        self.logs = DotMap()
        self.logs.state = []
        self.logs.action = []
        self.logs.traj_num = []

    def log_state(self, state, action):
        self.logs.action.append(action)
        self.logs.state.append(state)
        self.logs.traj_num.append(self.num_completed_trajectories)

    def set_torque(self, torques, log_data=True):
        """Send control signal to robot.

        :param torques: 1-D array of torques, torques.shape = [7]
        :return:
        """
        assert len(torques) == 7, "Torque must have 7 elements."
        torques = np.array(torques)
        torques[np.isinf(torques)] = 0
        torques[np.isnan(torques)] = 0
        bounds = np.array([10, 10, 10, 10, 10, 10, 10])
        torques = np.maximum(-bounds, np.minimum(bounds, torques))  # Bounds torques
        state = np.concatenate([joint_dict_to_arr(self._limb.joint_angles()),
                                joint_dict_to_arr(self._limb.joint_velocities())], axis=0)
        if log_data:
            self.log_state(state=state, action=torques)
        cmd = dict()
        for i, joint in enumerate(self._start_angles.keys()):
            # spring portion
            cmd[joint] = torques[i]
        self._limb.set_joint_torques(cmd)  # Send command to the robot

    def _update_parameters(self):
        for joint in self._limb.joint_names():
            self._springs[joint] = self._dyn.config[joint[-2:] + '_spring_stiffness']
            self._damping[joint] = self._dyn.config[joint[-2:] + '_damping_coefficient']

    def _update_forces(self, log_data=True):
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
            # Calculate joint-specific torque
            e = self.target[i] - cur_pos[joint]
            cmd[i] = self.P[i] * e - self.K[i] * cur_vel[joint]

        self.set_torque(cmd, log_data)  # command robot

    def collect_data(self, n_iters):
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

        unit_vector = np.array([0, 1, 0])  # Axis of rotation
        angle = 180 * np.pi / 180  # Angle of rotation
        mod_vec = unit_vector * np.sin(angle/2)
        quaternion = Quaternion(
            x=mod_vec[0], y=mod_vec[1], z=mod_vec[2],
            w=np.cos(angle/2)
        )

        # loop at specified rate commanding new joint torques
        breakout = False
        for _ in range(n_iters):
            # Generate a new goal
            self.target = None
            while self.target is None:
                if not self._rs.state().enabled:
                    rospy.logerr("Joint torque example failed to meet "
                                 "specified control rate timeout.")
                    breakout = True
                    break
                self.EE_target = [
                    np.random.uniform(0.6, 0.7),
                    np.random.uniform(-0.15, 0.15),
                    np.random.uniform(0.15, 0.45)
                ]
                res_dict = get_joint_angles(
                    get_pose(x=self.EE_target[0], y=self.EE_target[1], z=self.EE_target[2], o=quaternion),
                    use_advanced_options=False
                )
                if res_dict is not None:
                    self.target = joint_dict_to_arr(res_dict)
                else:
                    print("Unreachable target. Sending zero torques and recomputing.")
                    self.set_torque(np.zeros(shape=[7]), log_data=False)
                    control_rate.sleep()

            if breakout:
                break

            # Get to the goal
            while np.linalg.norm(self.target - joint_dict_to_arr(self._limb.joint_angles())) > 0.3 or \
                            np.linalg.norm(joint_dict_to_arr(self._limb.joint_velocities())) > 0.01:
                if not self._rs.state().enabled:
                    rospy.logerr("Joint torque example failed to meet "
                                 "specified control rate timeout.")
                    break
                self._update_forces()
                control_rate.sleep()

        self.num_completed_trajectories += 1

    def reset_to_initial_pos(self):
        """Moves robot back to initial position.

        :return: None
        """
        control_rate = rospy.Rate(self._freq)
        self.target = self.initial_angles
        while np.linalg.norm(self.target - joint_dict_to_arr(self._limb.joint_angles())) > 0.3 or \
                        np.linalg.norm(joint_dict_to_arr(self._limb.joint_velocities())) > 0.01:
            if not self._rs.state().enabled:
                rospy.logerr("Joint torque example failed to meet "
                             "specified control rate timeout.")
                break
            self._update_forces(log_data=False)
            control_rate.sleep()


    def clean_shutdown(self):
        """
        Switches out of joint torque mode to exit cleanly
        """
        # Save logs to file
        self.log_file.create_dataset("actions", data=np.array(self.logs.action))
        self.log_file.create_dataset("states", data=np.array(self.logs.state))
        self.log_file.create_dataset("trajectory_number", data=np.array(self.logs.traj_num), dtype='i')
        self.log_file.attrs['frequency'] = self._freq
        self.log_file.close()
        print('\nLogs saved to file: %s' % self.log_nameFile)

        print("Exiting example...")
        self._limb.exit_control_mode()


##################
# HELPER METHODS #
##################

def joint_dict_to_arr(joint_dict):
    return np.array([
        joint_dict['right_j6'],
        joint_dict['right_j5'],
        joint_dict['right_j4'],
        joint_dict['right_j3'],
        joint_dict['right_j2'],
        joint_dict['right_j1'],
        joint_dict['right_j0'],
    ])


if __name__ == "__main__":
    main()
