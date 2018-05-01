import grasp_net, grasp_params, h5py, aolib.img as ig, os, numpy as np, aolib.util as ut

net_pr = grasp_params.im_fulldata_v5()
net_pr = grasp_params.gel_im_fulldata_v5()
checkpoint_file = '/home/manu/ros_ws/src/manu_research/manu_sawyer/src/tensorflow_model_is_gripping/training/net.tf-6499'

gpu = '/gpu:0'
db_file = '/media/backup_disk/dataset_manu/ver2/2017-06-22/2017-06-22_212702.hdf5'

with h5py.File(db_file, 'r') as db:
    pre, mid, _ = grasp_net.milestone_frames(db)

    # sc = lambda x : ig.scale(x, (224, 224))
    def sc(x):
        """ do a center crop (helps with gelsight) """
        x = ig.scale(x, (256, 256))
        return ut.crop_center(x, 224)

    u = ig.uncompress
    crop = grasp_net.crop_kinect
    inputs = dict(
        gel0_pre=sc(u(db['GelSightA_image'].value[pre])),
        gel1_pre=sc(u(db['GelSightB_image'].value[pre])),
        gel0_post=sc(u(db['GelSightA_image'].value[mid])),
        gel1_post=sc(u(db['GelSightB_image'].value[mid])),
        im0_pre=sc(crop(u(db['color_image_KinectA'].value[pre]))),
        im0_post=sc(crop(u(db['color_image_KinectA'].value[mid]))),
        # these are probably unnecessary
        depth0_pre=sc(crop(db['depth_image_KinectA'].value[pre].astype('float32'))),
        depth0_post=sc(crop(db['depth_image_KinectA'].value[mid].astype('float32'))))

    net = grasp_net.NetClf(net_pr, checkpoint_file, gpu)
    prob = net.predict(**inputs)

    print 'prob = ', prob
