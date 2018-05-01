#!/usr/bin/env python

import rospy
from wsg_50_common.srv import *
from std_srvs.srv import *
from wsg_50_common.msg import Status, Cmd
import time
from rospy_message_converter import message_converter
import numpy as np

ErrorMessage = 'Gripper not connected, skipping gripper command:'

toRetry = False  # keep retry when motion failed


class WSG50(object):
    def __init__(self, speed=50):
        self.speed = speed
        self._bounds = np.array([1, 109])  # Limits of the gripper position [mm]
        self.subscribe_rostopic()
        self.homing()
        self.stateini()


    def stateini(self):
        ''' Due to some unknown bugs, calling this program first will make the system more stable'''
        self.graspmove_nopending(105, speed=0)
        time.sleep(0.2)
        self.graspmove_nopending(106, speed=0)
        time.sleep(0.2)
        self.graspmove_nopending(107, speed=0)
        time.sleep(0.2)
        self.graspmove_nopending(108, speed=0)

    def subscribe_rostopic(self):
        # rospy.init_node('qwe', anonymous=True)
        rospy.Subscriber("wsg_50_driver/status", Status)

    def move(self, pos=110, speed=None, toRetry=False):
        # move pos range: 0 - 110 mm
        # move speed range: 0- 420 mm/s

        if speed is None:
            speed = self.speed
        self.ack()
        command = 'move'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

        while True:
            try:
                error = srv(pos, speed)
                # print '[Gripper] move, return:', error
                break
            except:
                pass
                # print '[Gripper] move,', ErrorMessage, command

            if not toRetry: break
            time.sleep(0.5)

    def set_position(self, position, velocity=None):
        """
        Move to the objective position 'width', while stop if the force limit acquired.
        Return after the position reached
        :param position:
        :param velocity:
        :return:
        """
        if velocity is None:
            velocity = self.speed  # if velocity is not specified, use the default one.
        position = np.array(position)
        if np.isinf(position) or np.isnan(position):
            print('Invalid position')
        else:
            position = np.minimum(np.maximum(position, self._bounds[0]), self._bounds[1])
            # rospy.loginfo('Moving gripper to %f' % position)
            self.graspmove_nopending(position, velocity)
            Moving = True
            while Moving:
                # moving_state=rospy.wait_for_message("/wsg_50_driver/moving", std_msg.Bool)
                # Moving= moving_state.data
                gripper_state = rospy.wait_for_message("/wsg_50_driver/status", Status)
                if 'Stopped' in gripper_state.status or 'Target Pos reached' in gripper_state.status:
                    return

    def open(self, speed=None):
        if speed is None:
            speed = self.speed
        self.ack()
        # self.move(109, speed)
        self.set_position(position=self._bounds[1], velocity=speed)
        time.sleep(0.1)
        self.set_position(position=self._bounds[1], velocity=speed)
        time.sleep(0.1)
        self.set_position(position=self._bounds[1], velocity=speed)

    def close(self, speed=None):
        if speed is None:
            speed = self.speed
        self.ack()
        self.move(0, speed)

    def grasp(self, pos=5, speed=None):
        if speed is None:
            speed = self.speed
        self.ack()
        command = 'grasp'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

        while True:
            try:
                error = srv(pos, speed)
                # print '[Gripper] grasp, return:', error
                break
            except:
                pass
                # print '[Gripper] grasp,', ErrorMessage, command
            if not toRetry: break
            time.sleep(0.5)

    def graspmove_nopending(self, width, speed=20):
        '''move to the objective position 'width', while stop if the force limit acquired
           return before the position reached'''
        cmd = Cmd()
        cmd.mode = 'Script'
        cmd.pos = width
        cmd.speed = speed
        pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        pub.publish(cmd)

    def release(self, pos=110, speed=80):
        if speed is None:
            speed = self.speed
        self.ack()
        command = 'release'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Move)

        while True:
            try:
                error = srv(pos, speed)
                # print '[Gripper] release, return:', error
                break
            except:
                pass
                # print '[Gripper] release,', ErrorMessage, command
            if not toRetry: break
            time.sleep(0.5)

    def homing(self):
        self.ack()
        command = 'homing'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Empty)
        try:
            error = srv()
        except:
            pass
            # print '[Gripper] homing,', ErrorMessage, command

    def ack(self):
        command = 'ack'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Empty)
        try:
            error = srv()
        except:
            pass
            # print '[Gripper] ack,', ErrorMessage, command

    def set_force(self, val=5):
        # Range: 1 to 80 (N)
        command = 'set_force'
        srv = rospy.ServiceProxy('/wsg_50_driver/%s' % command, Conf)
        try:
            error = srv(val)
        except:
            pass
            # print '[Gripper] set_force,', ErrorMessage, command

    def get_force(self):
        message = rospy.wait_for_message("/wsg_50_driver/status", Status)
        state = message_converter.convert_ros_message_to_dictionary(message)
        force = state['force']
        return force

    def get_pos(self):
        message = rospy.wait_for_message("/wsg_50_driver/status", Status)
        state = message_converter.convert_ros_message_to_dictionary(message)
        force = state['width']
        return force

    def get_state(self):
        message = rospy.wait_for_message("/wsg_50_driver/status", Status)
        msg = message_converter.convert_ros_message_to_dictionary(message)
        state = np.array([msg['width'], msg['speed'], msg['acc'], msg['force']])
        return state
