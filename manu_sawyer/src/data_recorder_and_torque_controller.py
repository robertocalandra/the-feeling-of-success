#!/usr/bin/env python

import numpy as np
import h5py
import rospy
import time
# import tensorflow_model_is_gripping.aolib.img as ig, tensorflow_model_is_gripping.aolib.util as ut
import cv2
import KinectB_hd
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState


class DataRecorder_TorqueController:
    def __init__(self, limb, gripper, GelSightA, GelSightB, KinectA, KinectB, frequency=20):

        self.limb = limb
        self.des_angles = self.limb.joint_angles()  # record initial joint angles
        self.gripper = gripper

        self.KinectA = KinectA  # A is the one in front of the robot
        self.KinectB = KinectB  # B is the one above the robot
        self.KinectB_hd = KinectB_hd.KinectB_hd()
        time.sleep(0.1)

        # Classes for saving images from the GelSights
        self.GelSightA = GelSightA
        self.GelSightB = GelSightB

        # The directory where the data is going to get saved
        self.path = "/home/manu/ros_ws/src/manu_research/data/"

        self.namefile = []
        self.frequency = frequency
        self.missed_cmds = 5.0  # Missed cycles before triggering timeout
        self.recording = False
        self.file = []
        self.step_count = []
        self.data = dict()
        self.data['color_image_KinectA'] = []
        self.data['color_image_KinectB'] = []
        self.data['depth_image_KinectA'] = []
        self.data['depth_image_KinectB'] = []
        self.data['GelSightA_image'] = []
        self.data['GelSightB_image'] = []
        self.data['timestamp'] = []

        # self.P = np.array([25, 25, 30, 35, 45, 65, 45])
        # self.K = np.array([0.5, 0.5, 2, 3, 3, 5, 5])
        self.P = np.array([20, 20, 30, 50, 50, 200, 70])
        self.D = np.array([5, 5, 5, 5, 5, 5, 5])
        self.I = np.array([1, 1, 1, 1, 1, 1, 1])*5

        self.I_part = self.arr_to_joint_dict([0, 0, 0, 0, 0, 0, 0])

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + 'right' + '/suppress_cuff_interaction'
        self.pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        rospy.Subscriber("desired_joint_pos", JointState, self.set_des_angles)

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

    def arr_to_joint_dict(self, arr):
        return {
            'right_j6': arr[0],
            'right_j5': arr[1],
            'right_j4': arr[2],
            'right_j3': arr[3],
            'right_j2': arr[4],
            'right_j1': arr[5],
            'right_j0': arr[6],
        }

    def set_des_angles(self, jointstate):
        self.des_angles = dict(zip(jointstate.name, jointstate.position))

    def attach_springs(self):

        # set control rate
        control_and_sample_rate = rospy.Rate(self.frequency)

        # for safety purposes, set the control rate command timeout.
        # if the specified number of command cycles are missed, the robot
        # will timeout and disable
        self.limb.set_command_timeout((1.0 / self.frequency) * self.missed_cmds)

        while not rospy.is_shutdown():
            self.pub_cuff_disable.publish()

            torque = np.zeros(7)

            # record current angles/velocities
            cur_pos = self.limb.joint_angles()
            cur_vel = self.limb.joint_velocities()

            for i, joint in enumerate(self.des_angles.keys()):
                e = self.des_angles[joint] - cur_pos[joint]
                self.I_part[joint] = self.I_part[joint] * 0.95 + e
                torque[i] = self.P[i] * e
                torque[i] = torque[i] - self.D[i] * cur_vel[joint]
                torque[i] = torque[i] + self.I[i] * self.I_part[joint]

            self.set_torque(torque)

            if self.recording:
                self.save_data(torque)

            control_and_sample_rate.sleep()

    def set_torque(self, torques):
        torques = np.array(torques)
        torques[np.isinf(torques)] = 0
        torques[np.isnan(torques)] = 0
        bounds = np.array([20, 25, 35, 45, 60, 160, 100])
        torques = np.maximum(-bounds, np.minimum(bounds, torques))  # Bounds torques
        cmd = dict()
        for i, joint in enumerate(self.des_angles.keys()):
            # spring portion
            cmd[joint] = torques[i]
        self.limb.set_joint_torques(cmd)  # Send command to the rob

    def init_record(self, nameFile=None):
        """
        Initializes a recording session
        :param nameFile:
        """
        print('-- Data recording started --')

        # Pointer to file to write to
        self.namefile = self.path + nameFile + ".hdf5"
        self.file = h5py.File(self.namefile, "w")

        # For data structuring
        self.step_count = 0

        self.file.create_dataset("frequency", data=np.asarray([self.frequency]))

        # Save initial HD KinectB color image
        self.file.create_dataset("initial_hd_color_image_KinectB",
                                 data=self.KinectB_hd.get_color_image()[:, :, ::-1])

        self.data = dict()
        self.data['color_image_KinectA'] = []
        self.data['color_image_KinectB'] = []
        self.data['depth_image_KinectA'] = []
        self.data['depth_image_KinectB'] = []
        self.data['GelSightA_image'] = []
        self.data['GelSightB_image'] = []
        self.data['timestamp'] = []

        self.recording = True

    def save_data(self, torque):
        """
        Save state at the current instant to (hdf5) file
        """
        if self.recording:
            state_group = self.file.create_group("step_%012d" % self.step_count)

            # Get data
            curr_time = np.asarray([rospy.get_time()])
            limb_data = self.get_limb_data()
            gripper_data = self.gripper.get_state()
            gripper_names = ['width', 'speed', 'acc', 'force']
            endpoint_data = self.get_endeffector_data()
            endpoint_names = ["position", "orientation", "lin_velocity", "ang_velocity", "force_effort",
                              "torque_effort"]
            depth_image_KinectA = self.KinectA.get_depth_image()
            color_image_KinectA = self.KinectA.get_color_image()
            depth_image_KinectB = self.KinectB.get_depth_image()
            color_image_KinectB = self.KinectB.get_color_image()
            GelSightA_image = self.GelSightA.get_image()
            GelSightB_image = self.GelSightB.get_image()

            # Input data into file
            state_group.create_dataset("timestamp", data=curr_time)
            state_group.create_dataset("torque", data=torque)
            state_group.create_dataset("limb", data=limb_data)

            subgroup = state_group.create_group("gripper")
            for i in range(len(gripper_data)):
                subgroup.create_dataset(gripper_names[i], data=gripper_data[i])
            subgroup = state_group.create_group("endpoint")
            for i in range(len(endpoint_data)):
                subgroup.create_dataset(endpoint_names[i], data=endpoint_data[i])

            self.data['timestamp'].append(curr_time)
            # self.data['GelSightA_image'].append(GelSightA_image[:, :, ::-1])
            # self.data['GelSightB_image'].append(GelSightB_image[:, :, ::-1])
            # self.data['color_image_KinectA'].append(color_image_KinectA[:, :, ::-1])
            # self.data['color_image_KinectB'].append(color_image_KinectB[:, :, ::-1])
            self.data['GelSightA_image'].append(GelSightA_image)
            self.data['GelSightB_image'].append(GelSightB_image)
            self.data['color_image_KinectA'].append(color_image_KinectA)
            self.data['color_image_KinectB'].append(color_image_KinectB)
            self.data['depth_image_KinectA'].append(depth_image_KinectA)
            self.data['depth_image_KinectB'].append(depth_image_KinectB)
            self.step_count += 1

    def set_object_name(self, name):
        self.file.create_dataset("object_name", data=np.asarray([name]))

    def set_is_gripping(self, is_gripping):
        self.file.create_dataset("is_gripping", data=np.asarray([is_gripping]))

    def set_set_gripping_force(self, force):
        self.file.create_dataset("set_gripping_force", data=np.asarray([force]))

    def set_time_pre_grasping(self, time):
        self.file.create_dataset("time_pre_grasping", data=np.asarray([time]))

    def set_time_at_grasping(self, time):
        self.file.create_dataset("time_at_grasping", data=np.asarray([time]))

    def set_time_post1_grasping(self, time):
        self.file.create_dataset("time_post1_grasping", data=np.asarray([time]))

    def set_time_post2_grasping(self, time):
        self.file.create_dataset("time_post2_grasping", data=np.asarray([time]))

    def set_location_of_EE_at_grasping(self, loc):
        self.file.create_dataset("location_of_EE_at_grasping", data=np.asarray([loc]))

    def set_angle_of_EE_at_grasping(self, angle):
        self.file.create_dataset("angle_of_EE_at_grasping", data=np.asarray([angle]))

    def set_cylinder_data(self, xyz_kinect, height_object, radius):
        group = self.file.create_group("cylinder_data")
        group.create_dataset("xyz_kinect", data=xyz_kinect)
        group.create_dataset("height", data=height_object)
        group.create_dataset("radius", data=radius)

    def set_probability_A(self, probA):
        self.file.create_dataset("probability_A", data=np.asarray([probA]))

    def set_probability_B(self, probB):
        self.file.create_dataset("probability_B", data=np.asarray([probB]))

    def stop_record(self):
        print('-- Stopped recording data --')
        self.recording = False
        time.sleep(0.5)

        self.file.flush()
        self.file.close()
        self.file = None

    def export_to_file(self):

        # Pointer to file to write to
        self.file = h5py.File(self.namefile, "r+")

        time.sleep(0.1)

        ks = self.data.keys()

        map_fn = map

        steps = self.step_count

        print('-- Compressing images --')
        start_time = rospy.get_time()
        self.data['GelSightA_image'] = map_fn(compress_im, self.data['GelSightA_image'][:steps])
        self.file.create_dataset('GelSightA_image', data=self.data['GelSightA_image'])
        self.data['GelSightB_image'] = map_fn(compress_im, self.data['GelSightB_image'][:steps])
        self.file.create_dataset('GelSightB_image', data=self.data['GelSightB_image'])
        self.data['color_image_KinectA'] = map_fn(compress_im, self.data['color_image_KinectA'][:steps])
        self.file.create_dataset('color_image_KinectA', data=self.data['color_image_KinectA'])
        self.data['color_image_KinectB'] = map_fn(compress_im, self.data['color_image_KinectB'][:steps])
        self.file.create_dataset('color_image_KinectB', data=self.data['color_image_KinectB'])
        end_time = rospy.get_time()
        print("-- Compressing images -- Time:", end_time - start_time)

        self.file.create_dataset('timestamp', data=np.array(self.data['timestamp']))
        self.file.create_dataset('depth_image_KinectA', data=self.data['depth_image_KinectA'])
        self.file.create_dataset('depth_image_KinectB', data=self.data['depth_image_KinectB'])

        # Saves the total number of steps taken
        self.file.create_dataset("n_steps", data=np.asarray([self.step_count]))

        time.sleep(0.2)
        self.file.flush()
        self.file.close()  # Close file
        print('-- Saved file: %s --' % self.namefile)

        # Reset step_count
        self.step_count = 0

        for k in ks:
            self.data[k] = []

    def end_processes(self):
        """
        Stops treads
        """
        self.KinectA.end_process()
        self.KinectB.end_process()
        self.GelSightA.end_process()
        self.GelSightB.end_process()

    ####################################
    # Help functions used got get data #
    ####################################

    def get_endeffector_data(self):
        """
        Return EE state from Sawyer.
        Includes position, orientation, linear velocity, angular velocity, force and torque.
        """
        pose_dict = self.limb.endpoint_pose()
        velocity_dict = self.limb.endpoint_velocity()
        effort_dict = self.limb.endpoint_effort()

        pose_pos = np.asarray(pose_dict['position'])  # 3 numbers - x,y,z
        pose_orientation = np.asarray(pose_dict['orientation'])  # 4 numbers - x,y,z,w
        lin_velocity = np.asarray(velocity_dict['linear'])  # 3 numbers - x,y,z
        ang_velocity = np.asarray(velocity_dict['angular'])  # 3 numbers - x,y,z
        force_effort = np.asarray(effort_dict['force'])  # 3 numbers - x,y,z
        torque_effort = np.asarray(effort_dict['torque'])  # 3 numbers - x,y,z

        return [pose_pos, pose_orientation, lin_velocity, ang_velocity, force_effort, torque_effort]

    def get_limb_data(self):
        """
        Return state from the limb. Recall that Sawyer arm has 7 joints.
        Returns 3 x 7 array where each column corresponds to joint, each row angle, velocity and effort
        in alphabetical order.
        """
        angles = self.limb.joint_angles()  # dictionary
        velocities = self.limb.joint_velocities()  # dictionary
        efforts = self.limb.joint_efforts()  # dictionary

        angle_arr = []
        velocity_arr = []
        effort_arr = []
        for key, value in sorted(angles.items()):
            angle_arr.append(value)
        for key, value in sorted(velocities.items()):
            velocity_arr.append(value)
        for key, value in sorted(efforts.items()):
            effort_arr.append(value)

        return np.asarray([angle_arr, velocity_arr, effort_arr])


def compress_im(im):
    assert (im.dtype == np.uint8)
    return np.asarray(cv2.imencode('.jpg', im)[1].tostring())


def run_compress(x):
    x.export_to_file()
    return True


class CompressionTask:
    def __init__(self, recorder, pool):
        self.recorder = recorder
        self.pool = pool
        self.ret = None

    def run_async(self):
        self.ret = self.pool.map_async(run_compress, [self.recorder])

    def run_sync(self):
        run_compress(self.recorder)

    def wait(self):
        out = self.ret.get(int(1e6))
        self.ret = None
        return out
