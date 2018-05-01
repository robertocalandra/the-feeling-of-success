def write_tf():
  data = ut.load('../results/grasp-dset/v5/train.pk')
  tf_file = '../results/grasp-dset/v5/train.tf'
  assert not os.path.exists(tf_file)
  writer = tf.python_io.TFRecordWriter(tf_file)
  for d in ut.time_est(data):
    fbl = lambda x : tf.train.Feature(bytes_list = tf.train.BytesList(value = [x]))
    fl = lambda x : tf.train.Feature(float_list = tf.train.FloatList(value = map(float, x.flatten())))
    il = lambda x : tf.train.Feature(int64_list = tf.train.Int64List(value = x))

    feat = {'gel0_pre': fbl(d['gel0_pre']),
            'gel1_pre': fbl(d['gel1_pre']),
            'gel0_post': fbl(d['gel0_post']),
            'gel1_post': fbl(d['gel1_post']),
            'im0_pre': fbl(d['im0_pre']),
            'im0_post': fbl(d['im0_post']),
            'im1_pre': fbl(d['im1_pre']),
            'im1_post': fbl(d['im1_post']),
            'depth0_pre': fl(d['depth0_pre']),
            'depth0_post': fl(d['depth0_post']),
            'end_effector' : fl(d['end_effector']),
            'initial_press_prob' : fl(d['initial_press_prob']),
            'is_gripping' : il([d['is_gripping']])}
    ex = tf.train.Example(features = tf.train.Features(feature = feat))
    writer.write(ex.SerializeToString())
  writer.close()
  
