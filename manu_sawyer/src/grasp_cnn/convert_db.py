import os, sys, aolib.util as ut, aolib.img as ig, h5py

pj = ut.pjoin

in_file = sys.argv[1]
out_file = sys.argv[2]

convert_im = lambda x : ig.compress(x[:, :, ::-1], 'jpeg')

with h5py.File(in_file, 'r') as in_db:
  # os.system('cp %s %s' % (in_file, out_file))
  # print in_db.keys()

  # with h5py.File(out_file, 'r+') as out_db:
  #   im_names = ['GelSightA_image',
  #               'GelSightB_image',
  #               'color_image_KinectA',
  #               'color_image_KinectB']
  #   for name in im_names:
  #     out_db[name] = map(convert_im, in_db[name].value)


  with h5py.File(out_file, 'w') as out_db:
    vals = {}
    for k in sorted(in_db.keys()):
      if k.startswith('step_'):
        im_names = ['GelSightA_image',
                    'GelSightB_image',
                    'color_image_KinectA',
                    'color_image_KinectB']
        value_names = ['depth_image_KinectA', 
                       'depth_image_KinectB',
                       'timestamp']
        for name in im_names:
          ut.add_dict_list(vals, name, convert_im(in_db[k][name].value))
        for name in value_names:
          ut.add_dict_list(vals, name, in_db[k][name].value)
          
      else:
        #out_db.create_dataset(k, data = in_db[k].value if hasattr(in_db[k], 'value') else in_db[k])
        if hasattr(in_db[k], 'value'):
          out_db.create_dataset(k, data = in_db[k].value)
        else:
          print 'skipping:', k

    for name in vals:
      out_db.create_dataset(name, data=vals[name])

print 'Size before:'
os.system('du -ch %s' % in_file)
print 'Size after:'
os.system('du -ch %s' % out_file)
