#!/usr/bin/env python

import numpy as np
import h5py
import rospy
import time
import tensorflow_model_is_gripping.aolib.img as ig, tensorflow_model_is_gripping.aolib.util as ut
import cv2
import KinectB_hd


# Class for recording data from gripping experiment
class DataRecorder:
    def __init__(self, limb, gripper, GelSightA, GelSightB, KinectA, KinectB):
        """
        :param limb:
        :param gripper:
        """
        print('-- Initializing recorder --')

        self.limb = limb
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
        self.frequency = []
        self._recording = False
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

    def init_record(self, nameFile=None, frequency=10):
        """
        Initializes a recording session
        :param nameFile:
        :param frequency: recording frequency [Hz]
        """
        print('-- Data recording started --')
        self._recording = True

        # Pointer to file to write to
        self.namefile = self.path + nameFile + ".hdf5"
        self.file = h5py.File(self.namefile, "w")

        # For data structuring
        self.step_count = 0

        # Desired recording frequency
        self.frequency = frequency
        self.file.create_dataset("frequency", data=np.asarray([self.frequency]))

        # Save initial HD KinectB color image
        self.file.create_dataset("initial_hd_color_image_KinectB",
                                 data=self.KinectB_hd.get_color_image()[:, :, ::-1])

        # ks = self.data.keys()
        # for k in ks:
        #     self.data[k] = []

        # Starts to write to file
        self.continuously_record()

    def continuously_record(self):
        """
        Record data from the robot/cameras on a separate thread. Stop when told do stop.
        """
        rate = rospy.Rate(float(self.frequency))
        while self._recording:
            self.save_data()
            rate.sleep()

    def save_data(self):
        """
        Save state at the current instant to (hdf5) file
        """
        if self._recording:
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

        self._recording = False
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

        ks = self.data.keys()
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


def compress_data(d):
    d['GelSightA_image'] = compress_im(d['GelSightA_image'])
    d['GelSightB_image'] = compress_im(d['GelSightB_image'])
    d['color_image_KinectA'] = compress_im(d['color_image_KinectA'])
    d['color_image_KinectB'] = compress_im(d['color_image_KinectB'])
    return d


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
