# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
import tensorflow_model_is_gripping.aolib.img as ig, tensorflow_model_is_gripping.aolib.util as ut

__version__ = '0.2'
__author__ = 'Roberto Calandra'


def align_time(times):
    return times - times[0], times[0]


def delta_time(times):
    return times[1:-1] - times[0:-2]


def import_data(namefile):
    """
    Import a file into a formatted structure
    :param namefile:
    :return: data structure of type DotMap with the following fields:
        data.n_steps
        data.time
        data.kinect.A.image
        data.kinect.A.depth
        data.kinect.B.image
        data.kinect.B.depth
        data.gelsight.A.image
        data.gelsight.B.image

    """
    data = DotMap()
    data.n_steps = []
    data.frequency = []
    data.time = []
    data.robot.limb = []
    data.robot.gripper = []
    data.robot.EE = []  # TODO: not implemented yet
    data.kinect.A.image = []
    data.kinect.A.depth = []
    data.kinect.B.image = []
    data.kinect.B.depth = []
    data.gelsight.A.image = []
    data.gelsight.B.image = []
    data.kinect.B.hd.image = []

    print('Opening file: %s' % namefile)
    f = h5py.File(namefile, "r")
    data.n_steps = f['n_steps'].value
    print('Number of steps: %d' % data.n_steps)
    data.time = f['/timestamp'].value
    data.kinect.A.image = map(ig.uncompress, f['/color_image_KinectA'].value)
    data.kinect.A.depth = f['/depth_image_KinectA'].value
    data.kinect.B.image = map(ig.uncompress, f['/color_image_KinectB'].value)
    #decompress = lambda x : cv2.imdecode(np.asarray(bytearray(x), dtype = np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
    #data.kinect.B.image = map(decompress, f['/color_image_KinectB'].value)
    data.kinect.B.depth = f['/depth_image_KinectB'].value
    #cv2.imencode('.png', data.kinect.B.depth)

    # import lz4
    # ut.tic()
    # s = lz4.dumps(data.kinect.B.depth.tostring())
    # ut.toc('compress')
    # g = np.fromstring(lz4.loads(s), dtype = np.uint16)
    # g = g.reshape(data.kinect.B.depth.shape)
    # #print(g.dtype, g.shape)
    # #print(g)
    # assert((g == data.kinect.B.depth).all())
    # print('size before', ut.guess_bytes(data.kinect.B.depth))
    # print('size after', ut.guess_bytes(s))

    # import PIL.Image
    # import png
    # from StringIO import StringIO
    # im = data.kinect.B.depth[0, :, :, 0]
    # writer = png.Writer(width = im.shape[1], height = im.shape[0], bitdepth=16, compression = 3, greyscale = True)
    # io = StringIO()
    # ut.tic()
    # print(im.shape)
    # #writer.write(open('/tmp/test.png', 'w'), im)
    # writer.write(io, im)
    # ut.toc()
    # #u = io.getvalue()
    # import itertools
    # #pngdata = png.Reader(StringIO(io.getvalue())).read()[2]
    # #pngdata = png.Reader(StringIO(io.getvalue())).asDirect()[2]
    # #u = np.vstack(itertools.imap(np.uint16, pngdata))
    # #u = np.array(PIL.Image.open('/tmp/test.png'))
    # u = np.array(PIL.Image.open(StringIO(io.getvalue())))
    # #u = np.array(PIL.Image.open(StringIO(io.getvalue())), dtype = np.uint16)
    # #rint(u)
    # #print (u.shape, u.dtype, u[0])
    # assert((u == im).all())
    # print((u == im).all())
    # print('size before', ut.guess_bytes(im))
    # print('size after', ut.guess_bytes(io.getvalue()))

    # import PIL.Image
    # import png
    # from StringIO import StringIO
    # #im = data.kinect.B.depth[0, :, :, 0]
    # im = data.kinect.B.depth[..., 0]
    # im = im.reshape((im.shape[0]*im.shape[1], im.shape[2]))
    # writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16, compression=3, greyscale=True)
    # io = StringIO()
    # ut.tic()
    # writer.write(io, im)
    # ut.toc()
    # #
    # import itertools
    # u = np.array(PIL.Image.open(StringIO(io.getvalue())))
    # assert ((u == im).all())
    # print((u == im).all())
    # print('size before', ut.guess_bytes(im))
    # print('size after', ut.guess_bytes(io.getvalue()))
    #
    # return

    data.gelsight.A.image = map(ig.uncompress, f['/GelSightA_image'].value)
    data.gelsight.B.image = map(ig.uncompress, f['/GelSightB_image'].value)
    data.kinect.B.hd.image = f['/initial_hd_color_image_KinectB'].value

    plt.imshow(data.kinect.B.hd.image)


    # Convert to np arrays
    data.time = np.array(data.time)
    data.time, data.start_time = align_time(data.time)  # align time to first frame
    data.robot.limb = np.array(data.robot.limb)
    data.robot.gripper = np.array(data.robot.gripper)
    data.kinect.A.image = np.array(data.kinect.A.image)
    data.kinect.A.depth = np.array(data.kinect.A.depth).squeeze()
    data.kinect.B.image = np.array(data.kinect.B.image)
    data.kinect.B.depth = np.array(data.kinect.B.depth).squeeze()
    data.gelsight.A.image = np.array(data.gelsight.A.image)
    data.gelsight.B.image = np.array(data.gelsight.B.image)
    data.events = [f['/time_pre_grasping'].value, f['/time_at_grasping'].value, f['/time_post1_grasping'].value,
                   f['/time_post2_grasping'].value] - data.start_time
    print('Done')
    return data


def plot_state_limb(data):
    plt.figure()
    plt.plot(data.robot.limb)


def plot_gelsight_image(data, gelsight='A', id_frame=0):
    plt.figure()
    if gelsight == 'A':
        plt.imshow(data.gelsight.A.image[id_frame])
        plt.title('GelSight A (t=%d)' % id_frame)
    if gelsight == 'B':
        plt.imshow(data.gelsight.B.image[id_frame])
        plt.title('GelSight B (t=%d)' % id_frame)
    plt.axis('off')


def plot_kinect_depth(data, kinect='A', id_frame=0):
    """
    Plot image from the kinect depth
    :param data:
    :param kinect:
    :param id_frame:
    :return:
    """
    plt.figure()
    if kinect == 'A':
        plt.imshow(data.kinect.A.depth[id_frame], cmap='gray')
        plt.title('Kinect A (t=%d)' % id_frame)
    if kinect == 'B':
        plt.imshow(data.kinect.B.depth[id_frame], cmap='gray')
        plt.title('Kinect B (t=%d)' % id_frame)
    plt.axis('off')
    # plt.show()


def plot_kinect_image(data, kinect='A', id_frame=0):
    plt.figure()
    if kinect == 'A':
        plt.imshow(data.kinect.A.image[id_frame])
        plt.title('Kinect A (t=%d)' % id_frame)
    if kinect == 'B':
        plt.imshow(data.kinect.B.image[id_frame])
        plt.title('Kinect B (t=%d)' % id_frame)
    plt.axis('off')


def plot_frequency(data):
    plt.figure()
    # TODO: plot Groundtruth
    plt.hist(delta_time(data.time), 200)
    plt.xlabel('Delta time')


def plot_delta_time(data):
    plt.figure()
    plt.plot(delta_time(data.time))
    plt.ylabel('Delta Time')


def plot_events_time(data):
    plt.figure()
    plt.plot(data.time[2:], delta_time(data.time))
    plt.scatter(data.events, np.array([0, 0, 0, 0]))
    plt.ylabel('Time')


if __name__ == '__main__':
    nameFile = '/media/rcalandra/Data/datasets/gelsight/2017-06-06_1056PM_soft_red_cube_00000008.hdf5'
    nameFile = '/home/manu/ros_ws/src/manu_research/data/2017-06-20_162020.hdf5'
    nameFile = '/home/manu/ros_ws/src/manu_research/data/2017-06-20_162049.hdf5'
    nameFile = '/home/manu/ros_ws/src/manu_research/data/2017-06-20_195022.hdf5'
    nameFile = '/home/manu/ros_ws/src/manu_research/data/2017-07-06_164134.hdf5'
    data = import_data(namefile=nameFile)
    plot_state_limb(data=data)
    plot_kinect_depth(data=data, kinect='B', id_frame=5)
    plot_kinect_image(data=data, kinect='B', id_frame=5)
    plot_gelsight_image(data=data, gelsight='A', id_frame=5)
    plot_frequency(data)
    plot_delta_time(data)
    plot_events_time(data)
    plt.show()
