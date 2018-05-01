import numpy as np, tensorflow as tf, aolib.util as ut, aolib.img as ig, os, sys, sklearn.svm, press
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import vgg, h5py
import sklearn.metrics

pj = ut.pjoin

full_dim = 256
crop_dim = 224
#gpu = '/gpu:0'
init_path = '../results/vgg_16.ckpt'
#label_path = '../data/grasp'
label_path = '../data/grasp/labels'
checkpoint_iters = 1000
ed = tf.expand_dims
im_names = 'gel0_pre gel1_pre gel0_post gel1_post im0_pre im0_post depth0_pre depth0_post'.split()
write_data_gpu = 0
press_model_file = '../results/press-data-v11/training/net.tf-4600'
ee_dim = 4
 
def download_pretrained():
  # https://github.com/tensorflow/models/tree/master/slim
  ut.mkdir('../results')
  ut.sys_check('wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz '
               ' -O ../results/vgg_16_2016_08_28.tar.gz')
  ut.sys_check('cd ../results; tar -xzf vgg_16_2016_08_28.tar.gz')

def moving_avg(name, x, vals = {}, avg_win_size = 100):
  ut.add_dict_list(vals, name, x)
  return np.mean(vals[name][-avg_win_size:])


def shape(x, d = None):
  s = x.get_shape().as_list()
  return s if d is None else s[d]

def name_from_file(db_file):
  #return '_'.join(x.split('/')[-1].split('_')[2:])
  with h5py.File(db_file, 'r') as db:
    #print db.keys()
    name = str(np.array(db['object_name'].value)[0])
    name = remap_name(name)
    #print db_file, '->', name
    return name

def db_ok(db_file):
  try:
    with h5py.File(db_file, 'r') as db:
      ks = db.keys()
      reqs = ['GelSightA_image', 'GelSightB_image', 
              'is_gripping', 'angle_of_EE_at_grasping', 
              'location_of_EE_at_grasping', 'object_name', 
              'time_pre_grasping', 'time_post1_grasping',
              'color_image_KinectA', 'depth_image_KinectA',
              'timestamp']
      for r in reqs:
        if r not in ks:
          print db_file, 'is missing', r
          return False
      return True
  except:
    return False

def crop_kinect(im):
  # the bounds of the table, plus some padding above for tall objects
  bounds = np.array([[ 0.28602991, 0.07516428],
                     [ 0.76788474, 0.06441159],
                     [ 0.97554603, 0.59487754],
                     [ 0.13482023, 0.60563023]])
  d = np.array([im.shape[1], im.shape[0]])
  x0, y0 = map(int, bounds.min(0) * d)
  x1, y1 = map(int, bounds.max(0) * d)
  return im[y0 : y1, x0 : x1]

def milestone_frames(db):
  times = np.array(db['/timestamp'].value)
  pre = np.argmin(np.abs(db['time_pre_grasping'].value - times))
  mid = np.argmin(np.abs(db['time_at_grasping'].value - times))
  post = np.argmin(np.abs(db['time_post1_grasping'].value - times))
  return pre, mid, post

def remap_name(name):
  remap = {'plastic_duc': 'plastic_duck', 
           'hair_dryer_': 'hair_dryer_spiky_nozzle', 
           'light_blue_': 'light_blue_translucent_object', 
           'brown_paper': 'brown_paper_bag', 
           'beans_in_pa': 'beans_in_paper_container', 
           'shampoo_whi': 'shampoo_white_bottle', 
           'small_blue_': 'small_blue_plastic_spoon', 
           'dove_deodor': 'dove_deodorant', 
           'yellow_bulb': 'yellow_bulb_man',
           'toy_person_with_hat': 'set_small_plastic_men_yellow_construction_worker'}
  return remap.get(name, name)

#def write_data(out_dir, train_frac = 0.75, val_frac = 0.05):
#def write_data(out_dir, rebalance_data = False, train_frac = 0.75, val_frac = 0.0, n = None):
def write_data(out_dir, rebalance_data = True, train_frac = 0.75, val_frac = 0.0, n = None, seed = 0):
#def write_data(out_dir, rebalance_data = True, train_frac = 0.75, val_frac = 0.0, n = 10):
  assert not os.path.exists(out_dir)
  ut.mkdir(out_dir)
  base_data = '../data/grasp/'
  ut.sys_check('find -L %s -name "*.hdf5" > %s/all_db_files.txt' % (base_data, out_dir))

  all_db_files = ut.read_lines(pj(out_dir, 'all_db_files.txt'))[:n]
  all_db_files = ut.shuffled_with_seed(all_db_files, seed)
  all_db_files = filter(db_ok, all_db_files)
  ut.write_lines(pj(out_dir, 'db_files.txt'), all_db_files)

  by_name = ut.accum_dict((name_from_file(x), x) for x in all_db_files)

  names = ut.shuffled_with_seed(sorted(by_name.keys()), seed)
  num_names = len(names)
  num_train = int(train_frac * num_names)
  num_val = int(val_frac * num_names)
  i = 0
  train_names = names[i : num_train]
  i += num_train
  val_names = names[i : i + num_val]
  i += num_val
  test_names = names[i:]
  print num_train, num_val, len(test_names)

  splits = [('train', train_names),
            ('val', val_names),
            ('test', test_names)]

  print 'Number of objects in each split:'
  for s, o in splits:
    print s, '->', len(o)
  
  #press_clf = press.NetClf(press_model_file, gpu = write_data_gpu)
  press_clf = None#press.NetClf(press_model_file, gpu = write_data_gpu)
    
  for dset_name, names in splits:
    ut.write_lines(pj(out_dir, '%s_objects.txt' % dset_name), names)
    tf_file = pj(out_dir, '%s.tf' % dset_name)
    pk_file = pj(out_dir, '%s.pk' % dset_name)
    full_pk_file = pj(out_dir, 'full_%s.pk' % dset_name)

    if os.path.exists(tf_file):
      os.remove(tf_file)
    writer = tf.python_io.TFRecordWriter(tf_file)

    split_db_files = ut.flatten(by_name[name] for name in names)
    split_db_files = ut.shuffled_with_seed(split_db_files, dset_name)

    data = []
    for db_file in ut.time_est(split_db_files):
      with h5py.File(db_file, 'r') as db:
        #print 'keys =', db.keys()
        def im(x, crop = False, compress = True):
          x = ig.uncompress(x)
          x = np.array(x)
          if crop:
            x = crop_kinect(x)
            #ig.show(x)
          x = ig.scale(x, (256, 256), 1)
          if compress:
            x = ig.compress(x)
          return x

        def depth(x):
          x = np.array(x).astype('float32')
          x = ig.scale(x, (256, 256), 1)
          return x

        def parse_ee(x):
          names = ['angle_of_EE_at_grasping', 'location_of_EE_at_grasping']
          vs = [x[name].value for name in names]
          ee = np.concatenate([np.array(v).flatten() for v in vs]).astype('float32')
          return ee

        label_file = pj(label_path, db_file.split('/')[-1].replace('.hdf5', '.txt'))
        if os.path.exists(label_file):
          print 'Reading label from file'
          is_gripping = bool(ut.read_file(label_file))
        else:
          is_gripping = int(np.array(db['is_gripping']))

        pre, mid, _ = milestone_frames(db)

        # Estimate the probability that the robot is initially gripping the object
        if 0:
          press_a = press_clf.predict(
            im(db['/GelSightA_image'].value[mid], compress = False),
            im(db['/GelSightA_image'].value[pre], compress = False))
          press_b = press_clf.predict(
            im(db['/GelSightB_image'].value[mid], compress = False),
            im(db['/GelSightB_image'].value[pre], compress = False))
          initial_press_prob = 0.5 * (press_a + press_b)
        else:
          initial_press_prob = np.float32(-1.)
        #print initial_press_prob, ig.show(im(db['/GelSightA_image'].value[mid], compress = False))

        d = dict(gel0_pre = im(db['/GelSightA_image'].value[pre]),
                 gel1_pre = im(db['/GelSightB_image'].value[pre]),
                 gel0_post = im(db['/GelSightA_image'].value[mid]),
                 gel1_post = im(db['/GelSightB_image'].value[mid]),
                 
                 im0_pre = im(db['/color_image_KinectA'].value[pre], crop = True),
                 im0_post = im(db['/color_image_KinectA'].value[mid], crop = True),

                 im1_pre = im(db['/color_image_KinectB'].value[pre], crop = True),
                 im1_post = im(db['/color_image_KinectB'].value[mid], crop = True),
                 
                 depth0_pre = depth(crop_kinect(db['/depth_image_KinectA'].value[pre])),
                 depth0_post = depth(crop_kinect(db['/depth_image_KinectA'].value[mid])),

                 initial_press_prob = initial_press_prob,
                 is_gripping = int(is_gripping),
                 end_effector = parse_ee(db),
                 object_name = str(np.array(db['object_name'].value)[0]),

                 db_file = db_file)

        data.append(d)
    # for db files
    ut.save(full_pk_file, data)
    
    # rebalance data?
    if rebalance_data:
      by_label = [[], []]
      for x in ut.shuffled_with_seed(data, 'rebalance1'):
        by_label[x['is_gripping']].append(x)
      n = min(map(len, by_label))
      print len(data), 'before rebalance'
      data = ut.shuffled_with_seed(by_label[0][:n] + by_label[1][:n], 'rebalance2')
      print len(data), 'after rebalance'

    writer = tf.python_io.TFRecordWriter(tf_file)
    for d in data:
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

    ut.save(pk_file, data)
    print dset_name, '->', len(data), 'examples'

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
  

def read_example(rec_queue, pr):
  reader = tf.TFRecordReader()
  k, s = reader.read(rec_queue)
  feats = {'is_gripping' : tf.FixedLenFeature([], tf.int64),
           'end_effector' : tf.FixedLenFeature([3 + 1], tf.float32)}

  if 'gel' in pr.inputs:
    feats.update({'gel0_pre' : tf.FixedLenFeature([], dtype=tf.string),
                  'gel1_pre' : tf.FixedLenFeature([], dtype=tf.string),
                  'gel0_post' : tf.FixedLenFeature([], dtype=tf.string),
                  'gel1_post' : tf.FixedLenFeature([], dtype=tf.string)})
  if 'im' in pr.inputs:
    feats.update({'im0_pre' : tf.FixedLenFeature([], dtype=tf.string),
                  'im0_post' : tf.FixedLenFeature([], dtype=tf.string)})
  if 'depth' in pr.inputs:
    feats.update({'depth0_pre' : tf.FixedLenFeature([full_dim*full_dim], dtype=tf.float32),
                  'depth0_post' : tf.FixedLenFeature([full_dim*full_dim], dtype=tf.float32)})

  if 'ee' in pr.inputs:
    feats.update({'end_effector' : tf.FixedLenFeature([ee_dim], dtype=tf.float32)})

  example = tf.parse_single_example(s, features = feats)

  out = {'is_gripping' : example['is_gripping']}
  if 'ee' in pr.inputs:
    out['ee'] = example['end_effector']

  base_names = ['gel', 'im', 'depth']
  for base_name in base_names:
    if base_name not in pr.inputs:
      continue

    names = [name for name in im_names if name.startswith(base_name)]
    ims = []
    for name in names:
      im = example[name]
      if name.startswith('im') or name.startswith('gel'):
        im = tf.image.decode_png(im)
        im = tf.cast(im, tf.float32)
        im.set_shape((full_dim, full_dim, 3))
      elif name.startswith('depth'):
        im.set_shape((full_dim*full_dim))
        im = tf.reshape(im, (full_dim, full_dim))
        #im = tf.tile(ed(im, 2), (1, 1, 3))
      ims.append(im)

    combo = tf.concat(ims, 2)
    combo = tf.random_crop(combo, (crop_dim, crop_dim, shape(combo, 2)))
    combo = tf.image.random_flip_left_right(combo)
    if name.startswith('gel'):
      combo = tf.image.random_flip_up_down(combo)

    print 'group:'
    start = 0
    for name, im in zip(names, ims):
      out[name] = combo[:, :, start : start + shape(im, -1)]
      print name, shape(out[name])
      start += shape(im, -1)
    
  return out

def read_data(pr, num_gpus):
  tf_files = [pj(pr.dsdir, 'train.tf')]
  queue = tf.train.string_input_producer(tf_files)
  names, vals = ut.unzip(read_example(queue, pr).items())
  vals = tf.train.shuffle_batch(vals, batch_size = pr.batch_size,
                                capacity = 2000, min_after_dequeue = 500)
  if len(names) == 1:
    vals = [vals]

  splits = [{} for x in xrange(num_gpus)]
  for k, v in zip(names, vals):
    s = tf.split(v, num_gpus)
    for i in xrange(num_gpus):
      splits[i][k] = s[i]
  return splits

def normalize_ims(im):
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  return -1. + (2./255) * im 

def normalize_depth(depth):
  depth = tf.cast(depth, tf.float32)
  depth = depth / 1000.
  depth = -1. + (depth / 2.)
  return depth

def make_model(inputs, pr, train, reuse = False):
  n = normalize_ims
  def d(x): 
    x = normalize_depth(x)
    #x = tf.tile(ed(x, 2), (1, 1, 3))
    x = tf.tile(x, (1, 1, 1, 3))
    return x

  with slim.arg_scope(vgg.vgg_arg_scope(False)):
    feats = []
    if 'gel' in pr.inputs:
      if not hasattr(pr, 'gels') or (0 in pr.gels):
        print 'Using gel 0'
        gel0_pre, gel0_post = n(inputs['gel0_pre']), n(inputs['gel0_post'])
      else:
        gel0_pre, gel0_post = None, None

      if not hasattr(pr, 'gels') or (1 in pr.gels):
        print 'Using gel 1'
        gel1_pre, gel1_post = n(inputs['gel1_pre']), n(inputs['gel1_post'])
      else:
        gel1_pre, gel1_post = None, None

      if ut.hastrue(pr, 'gel_single'):
        print 'Using single gelsight model'
        if gel0_pre is not None:
          feats.append(vgg.vgg_16(gel0_post - gel0_pre, num_classes = None, 
                                  scope = 'gel_vgg16', is_training = train, reuse = reuse)[0])
        if gel1_pre is not None:
          feats.append(vgg.vgg_16(gel1_post - gel1_pre, num_classes = None, 
                                  scope = 'gel_vgg16', reuse = True, is_training = train)[0])
      else:
        print 'Using double gelsight model'
        feats.append(vgg.vgg_gel2(
          gel0_pre, gel0_post, 
          gel1_pre, gel1_post,
          is_training = train, 
          num_classes = None,
          reuse = reuse,
          scope = 'gel_vgg16'))

    if 'im' in pr.inputs:
      feats.append(vgg.vgg_16(n(inputs['im0_pre']), num_classes = None, 
                              scope = 'im_vgg16', is_training = train, reuse = reuse)[0])
      feats.append(vgg.vgg_16(n(inputs['im0_post']), num_classes = None, 
                              scope = 'im_vgg16', reuse = True, is_training = train)[0])

    if 'depth' in pr.inputs:
      feats.append(vgg.vgg_16(d(inputs['depth0_pre']), num_classes = None, 
                              scope = 'depth_vgg16', is_training = train, reuse = reuse)[0])
      feats.append(vgg.vgg_16(d(inputs['depth0_post']), num_classes = None, 
                              scope = 'depth_vgg16', is_training = train, reuse = True)[0])

    if 'ee' in pr.inputs:
      net = inputs['ee']
      net = net / ed(tf.sqrt(tf.reduce_sum(net**2, 1)), 1)
      net = slim.fully_connected(net, 4096, scope = 'ee/fc1', reuse = reuse)
      net = slim.fully_connected(net, 4096, scope = 'ee/fc2', reuse = reuse)
      net = slim.fully_connected(net, 4096, scope = 'ee/fc3', reuse = reuse)
      feats.append(net)

    net = tf.concat(feats, 1)
    logits = slim.fully_connected(net, 2, scope = 'logits', activation_fn = None, reuse = reuse)

  return logits

def gpu_strs(gpus):
  if gpus is not None and np.ndim(gpus) == 0:
    gpus = [gpus]
  return ['/cpu:0'] if gpus is None else ['/gpu:%d' % x for x in gpus]

def set_gpus(gpus):
  if gpus is None:
    return ['/cpu:0']
  else:
    if np.ndim(gpus) == 0:
      gpus = [gpus]
    os.putenv('CUDA_VISIBLE_DEVICES', ','.join(map(str, gpus)))
    gpus = range(len(gpus))
    return gpu_strs(gpus)

def average_grads(tower_grads):
  average_grads = []
  for ii, grad_and_vars in enumerate(zip(*tower_grads)):
    grads = []
    #print ii, len(grad_and_vars)
    for g, v in grad_and_vars:
      #print g, v.name
      if g is None:
        print 'skipping', v.name
        continue
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    if len(grads) == 0:
      #print 'no grads for', v.name
      grad = None
    else:
      #grad = tf.concat_v2(grads, 0)
      grad = tf.concat(grads, 0)
      #grad = mean_vals(grad, 0)
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train_press(pr):
  data = ut.load(pj(pr.dsdir, 'train.pk'))
  xs, ys = [], []
  for ex in data:
    xs.append([ex['initial_press_prob']])
    ys.append(ex['is_gripping'])
  xs = np.array(xs, 'float32')
  ys = np.array(ys, 'int64')

  clf = sklearn.svm.SVC(C = 1., kernel = 'linear')
  clf.fit(xs, ys)
  ut.save(pj(pr.resdir, 'clf.pk'), clf)

def train(pr, gpus, restore = False, use_reg = True):
  print 'Params:'
  print pr
  gpus = set_gpus(gpus)
  ut.mkdir(pr.resdir)
  ut.mkdir(pr.dsdir)
  ut.mkdir(pr.train_dir)
  config = tf.ConfigProto(allow_soft_placement = True)
  
  if pr.inputs == ['press']:
    return train_press(pr)

  with tf.Graph().as_default(), tf.device(gpus[0]), tf.Session(config = config) as sess:
    global_step = tf.get_variable('global_step', [], initializer = 
                              tf.constant_initializer(0), trainable = False)
    inputs = read_data(pr, len(gpus))
    lr = pr.base_lr * pr.lr_gamma**(global_step // pr.step_size)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
    if pr.opt_method == 'adam':
      opt = tf.train.AdamOptimizer(lr)
    elif pr.opt_method == 'momentum':
      opt = tf.train.MomentumOptimizer(lr, 0.9)

    gpu_grads = []
    for gi, gpu in enumerate(gpus):
      with tf.device(gpu):
        label = inputs[gi]['is_gripping']
        logits = make_model(inputs[gi], pr, train = True, reuse = (gi > 0))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits = logits, labels = label)
        loss = tf.reduce_mean(loss)
        if use_reg:
          reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
          print 'Number of regularization losses:', len(reg_losses)
          loss = loss + tf.add_n(reg_losses)
        eq = tf.equal(tf.argmax(logits, 1), label)
        acc = tf.reduce_mean(tf.cast(eq, tf.float32))
        gpu_grads.append(opt.compute_gradients(loss))
    #train_op = opt.minimize(loss, global_step = global_step)
    grads = average_grads(gpu_grads)
    train_op = opt.apply_gradients(grads, global_step = global_step)
    bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print 'Batch norm updates:', len(bn_ups)
    train_op = tf.group(train_op, *bn_ups)

    sess.run(tf.global_variables_initializer())
    var_list = slim.get_variables_to_restore()
    exclude = ['Adam', 'beta1_power', 'beta2_power', 'Momentum', 'global_step', 'logits', 'fc8', 'fc6_', 'fc7_', 'conv6']
    var_list = [x for x in var_list if \
                not any(name in x.name for name in exclude)]
    if restore:
      tf.train.Saver(var_list).restore(sess, tf.train.latest_checkpoint(pr.train_dir))
    else:
      #tf.train.Saver(var_list).restore(sess, init_path)

      for base in ['im', 'depth', 'gel']:
        print 'Restoring:', base
        mapping = {}
        for v in var_list:
          start = '%s_vgg16/' % base
          if v.name.startswith(start):
            vgg_name = v.name.replace(start, 'vgg_16/')
            vgg_name = vgg_name[:-2]
            print vgg_name, '->', v.name
            mapping[vgg_name] = v
        if len(mapping):
          tf.train.Saver(mapping).restore(sess, init_path)

    #saver = tf.train.Saver()
    tf.train.start_queue_runners(sess = sess)

    summary_dir = ut.mkdir('../results/summary')
    print 'tensorboard --logdir=%s' % summary_dir
    sum_writer = tf.summary.FileWriter(summary_dir, sess.graph)        
    for i in ut.time_est(range(pr.train_iters)):
      step = int(sess.run(global_step))
      if (step == 10 or step % checkpoint_iters == 0) or step == pr.train_iters - 1:
        check_path = pj(ut.mkdir(pr.train_dir), 'net.tf')
        print 'Saving:', check_path
        vs = slim.get_model_variables()
        tf.train.Saver(vs).save(sess, check_path, global_step = global_step)
      if step > pr.train_iters:
        break

      merged = tf.summary.merge_all()
      if step % 1 == 0:
        [summary] = sess.run([merged])
        sum_writer.add_summary(summary, step)      
      _, lr_val, loss_val, acc_val = sess.run([train_op, lr, loss, acc])

      if step % 10 == 0:
        print 'Iteration %d,' % step, 'lr = ', lr_val, \
              'loss:', moving_avg('loss', loss_val), 'acc:', moving_avg('acc', acc_val)
        sys.stdout.flush()
  
class NetClf:
  def __init__(self, pr, model_file, gpu = '/cpu:0'):
    self.sess = None
    self.pr = pr
    self.gpu = gpu
    self.model_file = model_file

  def __del__(self):
    self.deinit()
      
  def init(self):
    if self.sess is None:
      print 'Restoring:',self.model_file
      with tf.device(self.gpu):
        tf.reset_default_graph()
        print self.gpu
        tf.Graph().as_default()
        self.sess = tf.Session()
        s = (crop_dim, crop_dim, 3)
        self.gel0_pre = tf.placeholder(tf.uint8, s, name = 'gel0_pre')
        self.gel1_pre = tf.placeholder(tf.uint8, s, name = 'gel1_pre')
        self.gel0_post = tf.placeholder(tf.uint8, s, name = 'gel0_post')
        self.gel1_post = tf.placeholder(tf.uint8, s, name = 'gel1_post')
        self.im0_pre = tf.placeholder(tf.uint8, s, name = 'im0_pre')
        self.im0_post = tf.placeholder(tf.uint8, s, name = 'im0_post')
        self.depth0_pre = tf.placeholder(tf.float32, (crop_dim, crop_dim, 1), name = 'depth0_pre')
        self.depth0_post = tf.placeholder(tf.float32, (crop_dim, crop_dim, 1), name = 'depth0_post')
        self.ee = tf.placeholder(tf.float32, ee_dim, name = 'ee')
        inputs = {k : ed(getattr(self, k), 0) for k in im_names + ['ee']}
        # for k, v in inputs.items():
        #   print k, v.shape
        self.logits = make_model(inputs, self.pr, train = False)
        tf.train.Saver().restore(self.sess, self.model_file)
        tf.get_default_graph().finalize()
        
  def deinit(self):
    if self.sess is not None:
      self.sess.close()
      self.sess = None

  def format_im(self, im):
    return ig.scale(im, (crop_dim, crop_dim), 1)#.astype('float32')
    
  def predict(self, **kwargs):
    self.init()
    inputs = {}
    for k in im_names:
      inputs[getattr(self, k)] = self.format_im(kwargs[k])
    [logits] = self.sess.run([self.logits], inputs)
    print logits.shape
    p = ut.softmax(logits[0])[1]
    return (int(p >= 0.5), p)

class PressClf:
  """ A baseline that makes its decision based on whether the GelSight is pressed. """
  def __init__(self, pr):
    self.clf = ut.load(pj(pr.resdir, 'clf.pk'))

  def predict(self, **kwargs):
    d = self.clf.decision_function(np.array([[kwargs['initial_press_prob']]], dtype = 'float32'))
    return d >= 0, d

#def test(pr, gpu, test_on_train = False, center_crop = False):
def test(pr, gpu, test_on_train = False, center_crop = True):
  [gpu] = set_gpus([gpu])

  if pr.inputs == ['press']:
    net = PressClf(pr)
  else:
    #check_path = tf.train.latest_checkpoint(pr.train_dir)
    check_path = pj(pr.train_dir, 'net.tf-%d' % pr.model_iter)
    print 'Restoring from:', check_path
    net = NetClf(pr, check_path, gpu)

  if test_on_train:
    print 'Testing on train!'
    data = ut.load(pj(pr.dsdir, 'train.pk'))
  else:
    data = ut.load(pj(pr.dsdir, 'test.pk'))

  labels, probs, accs, vals = [], [], [], []
  for i in xrange(len(data)):
    ex = data[i]
    label = ex['is_gripping']
    def load_im(k, v):
      if k.startswith('gel') or k.startswith('im'):
        im = ig.uncompress(v)
      elif k.startswith('depth'):
        #v = np.tile(v, (1, 1, 3))
        im = v.astype('float32')
      else: 
        raise RuntimeError()
      if center_crop:
        im = ut.crop_center(im, 224)
      return im

    inputs = {k : load_im(k, ex[k]) for k in im_names}
    inputs['initial_press_prob'] = ex['initial_press_prob']
    inputs['ee'] = ex['end_effector']

    pred, prob = net.predict(**inputs)
    print prob, pred, label
    labels.append(label)
    probs.append(prob)
    accs.append(pred == label)
    print 'running average acc:', np.mean(accs)
    vals.append(ut.Struct(
      label = label,
      prob = prob,
      acc = accs[-1],
      idx = i,
      db_file = ex['db_file'],
      object_name = ex['object_name']))

  labels = np.array(labels, 'bool')
  probs = np.array(probs, 'float32')
  accs = np.array(accs)

  acc = np.mean(accs)
  ap = sklearn.metrics.average_precision_score(labels, probs)
  print 'Accuracy:', acc
  print 'mAP:', ap
  print 'Base rate:', ut.f3(np.array(ut.mapattr(vals).label).astype('float32').mean())

  ut.save(pj(pr.resdir, 'eval_results.pk'), 
          dict(acc = acc, ap = ap, 
               results = (labels, probs)))
  ut.save(pj(pr.resdir, 'eval.pk'), vals)

def color_depth(depth):
  depth = depth.astype('float32') / 1000.
  #import parula
  #return np.uint8(255*ut.apply_cmap(depth, parula.parula_map, 0.5, 1.2))
  return ut.clip_rescale_im(depth, 0.4, 1.)

def vis_example(db_file):
  with h5py.File(db_file, 'r') as db:
    pre, mid, post = milestone_frames(db)
    sc = lambda x : ig.scale(x, (600, None))
    im_mid = sc(crop_kinect(ig.uncompress(db['color_image_KinectA'][mid])))
    im_post = sc(crop_kinect(ig.uncompress(db['color_image_KinectA'][post])))
    depth = sc(color_depth(crop_kinect(db['depth_image_KinectA'][mid])))
    gel_a_0 = sc(ig.uncompress(db['GelSightA_image'][pre]))
    gel_b_0 = sc(ig.uncompress(db['GelSightB_image'][pre]))
    gel_a_1 = sc(ig.uncompress(db['GelSightA_image'][mid]))
    gel_b_1 = sc(ig.uncompress(db['GelSightB_image'][mid]))
    row = ['Color:', im_mid,
           'Depth:', depth,
           'Gel_A_1:', gel_a_1, 
           'Gel_B_1:', gel_b_1,
           'Gel_A_0:', gel_a_0, 
           'Gel_B_0:', gel_b_0,
           'Im after:', im_post,
           'Name:', str(np.array(db['object_name'].value[0])),
           'Path:', db_file.split('/')[-1]]
    return row

def analyze(pr):
  eval_exs = ut.load(pj(pr.resdir, 'eval.pk'))
  # accuracy by object
  by_name = ut.accum_dict((ex.object_name, ex) for ex in eval_exs)
  accs, labels = [], []
  for name in by_name:
    exs = by_name[name]
    accs.append(np.mean(ut.mapattr(exs).acc))
    labels.append(np.mean(ut.mapattr(exs).label))
    print name, ut.f4(accs[-1]), ut.f4(labels[-1])
  print 'Object-averaged accuracy:', ut.f4(np.mean(accs))
  print 'Object-averaged base:', ut.f4(np.mean(labels))

  chosen = set()
  table = []
  for ex in sorted(exs, key = lambda x : x.prob)[::-1]:
    if ex.object_name not in chosen:
      chosen.add(ex.object_name)
      print ex.object_name
      row = vis_example(ex.db_file)
      row = ['Prob:', ex.prob, 'Label:', ex.label] + row
      table.append(row)
  ig.show(table, rows_per_page = 25)

def show_db(pr, num_sample = None, num_per_object = 5):
  db_files = ut.read_lines(pj(pr.dsdir, 'db_files.txt'))
  db_files = ut.shuffled_with_seed(db_files)
  counts = {}
  db_files = ut.parfilter(db_ok, db_files)
  names = ut.parmap(name_from_file, db_files)
  table = []
  for name, db_file in zip(names, db_files[:num_sample]):
    if counts.get(name, 0) < num_per_object:
      counts[name] = 1 + counts.get(name, 0)
      row = vis_example(db_file)
      table.append(row)
  ig.show(table)

def get_object_names(pr):
  names = set()
  for x in ut.read_lines(pj(pr.dsdir, 'all_db_files.txt')):
    try:
      names.add(name_from_file(x))
    except:
      print 'Skipping:', x
  print '\n'.join(sorted(names))
  
def run(pr, todo = 'all', 
        gpu = 0, restore = 0):

  todo = ut.make_todo(todo, 'im train test')

  if 'im' in todo:
    write_data(pr.dsdir)

  if 'train' in todo:
    train(pr, gpu, restore = restore)

  if 'test' in todo:
    test(pr, gpu)

# def prepare_data():
#   for i in xrange(5):
#     path = '../results/dset/v1-split%d' % i
#     write_data(path)


def prepare(i):
  path = '../results/dset/v1-split%d' % i
  write_data(path, seed = 1 + i)

def prepare_data():
  #ut.parmap(prepare, range(5))
  map(prepare, range(5))

