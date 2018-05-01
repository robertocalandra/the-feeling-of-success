from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import rospy
from std_msgs.msg import Empty

import intera_interface
from intera_interface import CHECK_VERSION

from geometry_msgs.msg import Quaternion
from inverse_kin import get_joint_angles, get_pose


class SawyerEnv:
    def __init__(self, params):

        self._dyn = params.reconfig_server
        self._freq, self._missed_cmds = 20.0, 5.0

        # Control parameters
        self.bounds = params.bound * np.ones([7])

        # Create our limb instance
        self._limb = intera_interface.Limb(params.get('limb', 'right'))

        # Create cuff disable publisher
        cuff_ns = "robot/limb/%s/supress_cuff_interaction" % params.get('limb', 'right')
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        print("Running. Ctrl-c to quit")

        # PD controller gains for resets
        self.P = np.array([5, 5, 10, 15, 30, 20, 25])             # P controller gains
        self.K = np.array([0.7, 0.2, 1.63, 0.5, 3.5, 4.7, 6.0])   # D controller gains

        self._smoother_params = params.smoother_params
        self._smoother_history = np.zeros(shape=[self._smoother_params.get("history_length", 1), 7])

        self.sent_torques = []

    def set_command_timeout(self, t):
        self._limb.set_command_timeout(t)

    def step(self, a):
        self._pub_cuff_disable.publish()

        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet specified control rate timeout.")
            return self._get_obs(), 0, True, dict()
        self._set_torque(a)

        ob = self._get_obs()
        reward = 0  # Ignore reward
        done = False  # Never done except for errors
        return ob, reward, done, dict()

    def reset(self):
        self._move_to_rand_pos()
        return self._get_obs()

    def get_torque_log(self):
        return self.sent_torques

    def reset_torque_log(self):
        self.sent_torques = []

    def clean_shutdown(self):
        self._limb.exit_control_mode()

    def _get_obs(self):
        """Returns the current observation

        :return:
        """
        angles = self._joint_dict_to_arr(self._limb.joint_angles())
        velocities = self._joint_dict_to_arr(self._limb.joint_velocities())
        return np.concatenate([angles, velocities], axis=0)

    def _set_torque(self, torques, log_torque=True, smooth_torque=True):
        """Preprocess and send control signal to the Sawyer

        :param torques: Torques to be sent to the Sawyer
        :param log_data: bool indicating whether or not to log data.
        :return: None
        """
        assert len(torques) == 7, "Torque must have 7 elements"
        torques = np.array(torques)
        torques[np.isinf(torques)], torques[np.isnan(torques)] = 0, 0
        torques = np.maximum(-self.bounds, np.minimum(self.bounds, torques))  # Bounds torques

        if smooth_torque and self._smoother_params.smooth_torques:
            torques = self._smooth_torques(torques)
        if log_torque:
            self.sent_torques.append(torques)

        self._limb.set_joint_torques(self._arr_to_joint_dict(torques))

    def _move_to_rand_pos(self):
        """Moves the end-effector to a random position.

        :return: None
        """
        control_rate = rospy.Rate(self._freq)

        angle_target = None
        unit_vector = np.array([0, 1, 0])  # Axis of rotation
        angle = 180 * np.pi / 180  # Angle of rotation
        mod_vec = unit_vector * np.sin(angle/2)
        quaternion = Quaternion(
            x=mod_vec[0], y=mod_vec[1], z=mod_vec[2],
            w=np.cos(angle/2)
        )

        control_rate = rospy.Rate(self._freq)
        self.set_command_timeout((1.0 / self._freq) * self._missed_cmds)

        # Setting the target

        while angle_target is None:
            ee_target = [np.random.uniform(0.6, 0.7), np.random.uniform(-0.15, 0.15), np.random.uniform(0.25, 0.45)]
            res_dict = get_joint_angles(
                get_pose(x=ee_target[0], y=ee_target[1], z=ee_target[2], o=quaternion),
                use_advanced_options=False
            )
            if res_dict is not None:
                angle_target = self._joint_dict_to_arr(res_dict)
            else:
                print("Unreachable target. Sending zero torques and recomputing.")
                self._limb.set_joint_torques(self._arr_to_joint_dict(np.zeros(shape=[7])))
                control_rate.sleep()

        curr_bound = np.copy(self.bounds)
        self.bounds = 9 * np.ones([7])

        # PD Controller

        while np.linalg.norm(angle_target - self._joint_dict_to_arr(self._limb.joint_angles())) > 0.3 or \
                        np.linalg.norm(self._joint_dict_to_arr(self._limb.joint_velocities())) > 0.01:
            if not self._rs.state().enabled:
                rospy.logerr("Joint torque example failed to meet "
                             "specified control rate timeout.")
                break

            self._pub_cuff_disable.publish()

            if not self._rs.state().enabled:
                return

            cmd = np.zeros(7)
            cur_pos = self._joint_dict_to_arr(self._limb.joint_angles())
            cur_vel = self._joint_dict_to_arr(self._limb.joint_velocities())
            for i in range(7):
                e = angle_target[i] - cur_pos[i]
                cmd[i] = self.P[i] * e - self.K[i] * cur_vel[i]

            self._set_torque(cmd, log_torque=False, smooth_torque=False)
            control_rate.sleep()

        self.bounds = curr_bound

    def _smooth_torques(self, torque):
        self._smoother_history = np.concatenate([np.array([torque]), self._smoother_history], axis=0)[:-1]
        return np.mean(self._smoother_history, axis=0)

    @staticmethod
    def _joint_dict_to_arr(joint_dict):
        """Converts from a dictionary representation of the joints
        to an array.

        :param joint_dict: The state of the limb as a dictionary.
        :return: The converted state.
        """
        return np.array([
            joint_dict['right_j6'],
            joint_dict['right_j5'],
            joint_dict['right_j4'],
            joint_dict['right_j3'],
            joint_dict['right_j2'],
            joint_dict['right_j1'],
            joint_dict['right_j0'],
        ])

    @staticmethod
    def _arr_to_joint_dict(arr):
        """Converts from an array representation of the joints
        to a dict.

        :param arr: The state of the limb as an array.
        :return: The converted state.
        """
        return {
            'right_j6': arr[0],
            'right_j5': arr[1],
            'right_j4': arr[2],
            'right_j3': arr[3],
            'right_j2': arr[4],
            'right_j1': arr[5],
            'right_j0': arr[6],
        }
