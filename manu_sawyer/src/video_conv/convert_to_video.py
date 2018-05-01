# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import

# ----------------------------------------------------------------------------------------------------------------------

from dotmap import DotMap
import numpy as np
import h5py
import sys
sys.path.insert(0, '/home/manu/ros_ws/src/Research/manu_sawyer/src/tensorflow_model_is_gripping')
import aolib.img as ig
import RGB2video as RGB2video


def convert_to_video(path, list_filenames):

    if isinstance(list_filenames, basestring):
        list_filenames = [list_filenames]  # only a string is convert to a list

    for file in list_filenames:

        namefile = path + file

        data = DotMap()
        data.n_steps = []
        data.frequency = []
        data.kinect.A.image = []
        data.gelsight.A.image = []
        data.gelsight.B.image = []

        print('Opening file: %s' % namefile)
        f = h5py.File(namefile, "r")
        data.n_steps = f['n_steps'].value
        print('Number of steps: %d' % data.n_steps)
        data.frequency = int(f['frequency'].value)
        print('FPS: %d' % data.frequency)

        data.kinect.A.image = map(ig.uncompress, f['/color_image_KinectA'].value)
        print("color_image_KinectA done")

        data.gelsight.A.image = map(ig.uncompress, f['/GelSightA_image'].value)
        print("GelSightA_image done")

        data.gelsight.B.image = map(ig.uncompress, f['/GelSightB_image'].value)
        print("GelSightB_image done")

        # Convert to np arrays
        kinect_A = np.asarray(data.kinect.A.image)
        print('kinect.A To array done')
        gelsight_A = np.asarray(data.gelsight.A.image)
        print('gelsight.A To array done')
        gelsight_B = np.asarray(data.gelsight.B.image)
        print('gelsight.B To array done')

        print(kinect_A.shape)
        print(gelsight_A.shape)
        print(gelsight_B.shape)

        print(RGB2video.RGB2video(data=kinect_A, nameFile=file + '_kinect_A', framerate=data.frequency))
        print(RGB2video.RGB2video(data=gelsight_A, nameFile=file + '_gelsight_A', framerate=data.frequency))
        print(RGB2video.RGB2video(data=gelsight_B, nameFile=file + '_gelsight_B', framerate=data.frequency))

if __name__ == '__main__':
    path = '/home/manu/ros_ws/src/Research/manu_sawyer/src/video_conv/'
    import os

    list_filenames = []
    for file in os.listdir(path):
        if file.endswith(".hdf5"):
            list_filenames.append(file)
    list_filenames = sorted(list_filenames)

    convert_to_video(path=path, list_filenames=list_filenames)
