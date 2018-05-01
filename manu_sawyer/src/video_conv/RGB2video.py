# import R.log as rlog
import matplotlib.pyplot as plt
import numpy as np

# TODO: use multiprocessing
# from multiprocessing import Pool


def RGB2video(data, nameFile='video', framerate=24, codec='mpeg4', threads=4, verbosity=1, indent=0):
    """
    Encode and save RGB data as a video file.
    Requirements: moviepy
    :param data: np.array N x H x W x 3, where H=height, W=width, and N is the number of frames
    :param nameFile: name of the file to be saved
    :param framerate: scalar>0.
    :param codec: string.
    :param threads: scalar >0.
    :param verbosity: scalar.
    :param indent: scalar.
    :return:
    """
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter as fwv

    assert framerate > 0, 'framerate must be >0'
    assert threads > 0, 'threads must be >0'
    assert data.ndim == 4, 'Incorrect shape of data'
    # assert data.shape(3) == 3, 'Incorrect shape of data'

    try:
        # Write to FFMPEG
        extension = '.mp4'  # '.avi'
        fullNameVideo = nameFile + extension
        n_frame = data.shape[0]
        resolution = (data.shape[2], data.shape[1])  # (W, H)
        # TODO: use rlog.cnd_msg(), after fixing import rlog
        print('Resolution: %d x %d fps: %d n_frames: %d' % (resolution[0], resolution[1], framerate, n_frame))
        print('Saving to file: ' + fullNameVideo)
        a = fwv(filename=fullNameVideo, codec=codec, size=resolution, fps=framerate, preset="slower", threads=threads)
        for i in range(n_frame):
            # frame = np.swapaxes(data[i, :], 1, 2)
            frame = data[i, :].astype('uint8')  # Convert to uint8
            assert np.all(0 <= frame) and np.all(frame <= 255), 'Value of the pixels is not in [0-255]'  # Check data
            a.write_frame(frame)  # Write to file
            # plt.figure()
            # plt.imshow(frame/255)
            # plt.show
        a.close()  # Close file
        status = 0
        # rlog.cnd_status(current_verbosity=verbosity, necessary_verbosity=1, f=0)
        # TODO: fix circular: import rlog
    except:
        print('Something failed')
        status = 1

    return status
