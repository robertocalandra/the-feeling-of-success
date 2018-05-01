import numpy as np, tensorflow as tf, aolib.util as ut, aolib.img as ig, os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import vgg, h5py
import sklearn.metrics

pj = ut.pjoin

full_dim = 256
crop_dim = 224
train_iters = 10000
batch_size = 32
#batch_size = 32
#base_lr = 1e-4
base_lr = 1e-4
gamma = 0.5
step_size = 1000
sample_dur_secs = 0.2
sample_fps = 60
gpu = '/gpu:0'
init_path = '../results/vgg_16.ckpt'
checkpoint_iters = 100
update_top_only = False

ed = tf.expand_dims
im_names = 'gel0_pre gel1_pre gel0_post gel1_post'.split()

def download_pretrained():
  # https://github.com/tensorflow/models/tree/master/slim
  ut.mkdir('../results')
  ut.sys_check('wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz '
               ' -O ../results/vgg_16_2016_08_28.tar.gz')
  ut.sys_check('cd ../results; tar -xzf vgg_16_2016_08_28.tar.gz')

def moving_avg(name, x, vals = {}, avg_win_size = 100):
  ut.add_dict_list(vals, name, x)
  return np.mean(vals[name][-avg_win_size:])

def read_example(rec_queue):
  reader = tf.TFRecordReader()
  k, s = reader.read(rec_queue)
  feats = {
    'gel0_pre' : tf.FixedLenFeature([], dtype=tf.string),
    'gel1_pre' : tf.FixedLenFeature([], dtype=tf.string),
    'gel0_post' : tf.FixedLenFeature([], dtype=tf.string),
    'gel1_post' : tf.FixedLenFeature([], dtype=tf.string),
    'is_gripping' : tf.FixedLenFeature([], tf.int64),
    }
  example = tf.parse_single_example(s, features = feats)
  gel0_pre = tf.image.decode_png(example['gel0_pre'])
  gel1_pre = tf.image.decode_png(example['gel1_pre'])
  gel0_post = tf.image.decode_png(example['gel0_post'])
  gel1_post = tf.image.decode_png(example['gel1_post'])
  
  ims = [gel0_pre, gel1_pre, gel0_post, gel1_post]
  for x in ims:
    x.set_shape((full_dim, full_dim, 3))
  
  combo = tf.concat([ed(x, 0) for x in ims], 0)
  combo = tf.random_crop(combo, (shape(combo, 0), crop_dim, 
                                 crop_dim, shape(combo, 3)))
  gel0_pre = combo[0]
  gel1_pre = combo[1]
  gel0_post = combo[2]
  gel1_post = combo[3]

  label = example['is_gripping']

  return gel0_pre, gel1_pre, gel0_post, gel1_post, label

def read_data(path):
  tf_files = [pj(path, 'train.tf')]
  queue = tf.train.string_input_producer(tf_files)
  gel0_pre, gel1_pre, gel0_post, gel1_post, labels =  \
            tf.train.shuffle_batch(read_example(queue),
                                   batch_size = batch_size,
                                   capacity = 2000, min_after_dequeue = 500)
  return dict(gel0_pre = gel0_pre, 
              gel1_pre = gel1_pre, 
              gel0_post = gel0_post, 
              gel1_post = gel1_post), labels

def normalize_ims(im):
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  return -1. + (2./255) * im 

def shape(x, d = None):
  s = x.get_shape().as_list()
  return s if d is None else s[d]

def write_data(out_dir, train_frac = 0.75, val_frac = 0.05):
  ut.mkdir(out_dir)
  base_data = '../data/grasp/'
  ut.sys_check('find %s -name "*.hdf5" > %s/db_files.txt' % (base_data, out_dir))


  all_db_files = ut.read_lines(pj(out_dir, 'db_files.txt'))
  all_db_files = ut.shuffled_with_seed(all_db_files)
  name_from_file = lambda x : '_'.join(x.split('/')[-1].split('_')[2:])

  by_name = ut.accum_dict((name_from_file(x), x) for x in all_db_files)

  names = ut.shuffled_with_seed(sorted(by_name.keys()))
  num_names = len(all_db_files)
  num_train = int(train_frac * num_names)
  num_val = int(val_frac * num_names)
  i = 0
  train_names = names[i : num_train]
  i += num_train
  val_names = names[i : i + num_val]
  i += num_val
  test_names = names[i:]

  for dset_name, names in [('train', train_names),
                           ('val', val_names),
                           ('test', test_names)]:
    ut.write_lines(pj(out_dir, '%s_objects.txt' % dset_name), names)
    tf_file = pj(out_dir, '%s.tf' % dset_name)
    pk_file = pj(out_dir, '%s.pk' % dset_name)


    if os.path.exists(tf_file):
      os.remove(tf_file)
    writer = tf.python_io.TFRecordWriter(tf_file)

    data = []
    for name in names:
      for db_file in by_name[name]:
        with h5py.File(db_file, 'r') as db:
          def im(x):
            x = np.array(x)
            x = ig.scale(x, (256, 256), 1)
            return ig.compress(x)

          if 'is_gripping' in db:
            label = int(np.array(db['is_gripping']))
          elif 'Is gripping?' in db:
            label = int(np.array(db['Is gripping?']))
          else:
            print 'Skipping: %s. Missing is_gripping' % db_file
            print 'Keys:', ' '.join(db.keys())
            continue

          data.append({
            'gel0_pre': im(db['GelSightA_image_pre_gripping']),
            'gel1_pre': im(db['GelSightB_image_pre_gripping']),
            'gel0_post': im(db['GelSightA_image_post_gripping']),
            'gel1_post': im(db['GelSightB_image_post_gripping']),
            'is_gripping' : label})

          fbl = lambda x :tf.train.Feature(bytes_list = tf.train.BytesList(value = [x]))
          feat = {
            'gel0_pre': fbl(im(db['GelSightA_image_pre_gripping'])),
            'gel1_pre': fbl(im(db['GelSightB_image_pre_gripping'])),
            'gel0_post': fbl(im(db['GelSightA_image_post_gripping'])),
            'gel1_post': fbl(im(db['GelSightB_image_post_gripping'])),
            'is_gripping' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))}
          ex = tf.train.Example(features = tf.train.Features(feature = feat))
          writer.write(ex.SerializeToString())
    writer.close()
    ut.save(pk_file, data)
    print dset_name, '->', len(data), 'examples'

def make_model(inputs, train):
  n = normalize_ims
  logits = vgg.vgg_gel2(
    n(inputs['gel0_pre']), n(inputs['gel0_post']), 
    n(inputs['gel1_pre']), n(inputs['gel1_post']), 
    is_training = train, 
    update_top_only = update_top_only, 
    num_classes = 2)
  return logits

# def train(path, restore = False):
#   config = tf.ConfigProto(allow_soft_placement = True)
#   with tf.Graph().as_default(), tf.device(gpu), tf.Session(config = config) as sess:
#     global_step = tf.get_variable(
#       'global_step', [], initializer = 
#       tf.constant_initializer(0), trainable = False)
#     inputs, labels = read_data(path)
#     logits = make_model(inputs, train = True)
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#       logits = logits, labels = labels)
#     loss = tf.reduce_mean(loss)
#     tf.summary.scalar('loss', loss)
#     eq = tf.equal(tf.argmax(logits, 1), labels)
#     acc = tf.reduce_mean(tf.cast(eq, tf.float32))
#     tf.summary.scalar('acc', acc)

#     lr = base_lr * gamma**(global_step // step_size)
#     opt = tf.train.MomentumOptimizer(lr, 0.9)
#     train_op = opt.minimize(loss, global_step = global_step)
#     bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     print 'Batch norm updates:', len(bn_ups)
#     train_op = tf.group(train_op, *bn_ups)

#     sess.run(tf.global_variables_initializer())
#     var_list = slim.get_variables_to_restore()
#     exclude = ['Adam', 'beta1_power', 'beta2_power', 'Momentum', 'global_step', 'logits', 'fc8']
#     var_list = [x for x in var_list if \
#                 not any(name in x.name for name in exclude)]
#     train_dir = pj(path, 'training')
#     if restore:
#       tf.train.Saver(var_list).restore(sess, tf.train.latest_checkpoint(train_dir))
#     else:
#       tf.train.Saver(var_list).restore(sess, init_path)
#     #saver = tf.train.Saver()
#     tf.train.start_queue_runners(sess = sess)

#     summary_dir = ut.mkdir('../results/summary')
#     print 'tensorboard --logdir=%s' % summary_dir
#     sum_writer = tf.summary.FileWriter(summary_dir, sess.graph)        
#     while True:
#       step = int(sess.run(global_step))
#       if (step == 10 or step % checkpoint_iters == 0) or step == train_iters - 1:
#         check_path = pj(ut.mkdir(train_dir), 'net.tf')
#         print 'Saving:', check_path
#         vs = slim.get_model_variables()
#         tf.train.Saver(vs).save(sess, check_path, 
#                                 global_step = global_step)
#       if step > train_iters:
#         break

#       merged = tf.summary.merge_all()
#       if step % 1 == 0:
#         [summary] = sess.run([merged])
#         sum_writer.add_summary(summary, step)      
#       _, lr_val, loss_val, acc_val = sess.run([train_op, lr, loss, acc])

#       if step % 10 == 0:
#         print 'Iteration', step, 'lr = ', lr_val, \
#               'loss:', loss_val, 'acc:', acc_val


def train(path, restore = False):
  config = tf.ConfigProto(allow_soft_placement = True)
  with tf.Graph().as_default(), tf.device(gpu), tf.Session(config = config) as sess:
    global_step = tf.get_variable('global_step', [], initializer = 
                              tf.constant_initializer(0), trainable = False)
    inputs, labels = read_data(path)
    logits = make_model(inputs, train = True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(loss)
    eq = tf.equal(tf.argmax(logits, 1), labels)
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))

    lr = base_lr * gamma**(global_step // step_size)
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    train_op = opt.minimize(loss, global_step = global_step)
    bn_ups = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print 'Batch norm updates:', len(bn_ups)
    train_op = tf.group(train_op, *bn_ups)

    sess.run(tf.global_variables_initializer())
    var_list = slim.get_variables_to_restore()
    exclude = ['Adam', 'beta1_power', 'beta2_power', 'Momentum', 'global_step', 'logits', 'fc8', 'fc6_', 'fc7_', 'conv6']
    var_list = [x for x in var_list if \
                not any(name in x.name for name in exclude)]
    train_dir = pj(path, 'training')
    if restore:
      tf.train.Saver(var_list).restore(sess, tf.train.latest_checkpoint(train_dir))
    else:
      tf.train.Saver(var_list).restore(sess, init_path)
    #saver = tf.train.Saver()
    tf.train.start_queue_runners(sess = sess)

    summary_dir = ut.mkdir('../results/summary')
    print 'tensorboard --logdir=%s' % summary_dir
    sum_writer = tf.summary.FileWriter(summary_dir, sess.graph)        
    while True:
      step = int(sess.run(global_step))
      if (step == 10 or step % checkpoint_iters == 0) or step == train_iters - 1:
        check_path = pj(ut.mkdir(train_dir), 'net.tf')
        print 'Saving:', check_path
        #saver.save(sess, check_path, global_step = global_step)
        vs = slim.get_model_variables()
        # print 'Variables:'
        # for x in vs:
        #   print x.name
        tf.train.Saver(vs).save(sess, check_path, global_step = global_step)
      if step > train_iters:
        break

      merged = tf.summary.merge_all()
      if step % 1 == 0:
        [summary] = sess.run([merged])
        sum_writer.add_summary(summary, step)      
      _, lr_val, loss_val, acc_val = sess.run([train_op, lr, loss, acc])

      if step % 10 == 0:
        print 'Iteration', step, 'lr = ', lr_val, 'loss:', moving_avg('loss', loss_val), 'acc:', moving_avg('acc', acc_val)
  
class NetClf:
  def __init__(self, model_file, gpu = '/cpu:0'):
    self.sess = None
    self.model_file = model_file
    self.gpu = gpu

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
        self.logits = make_model({k : ed(getattr(self, k), 0) for k in im_names}, train = False)
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
    return ut.softmax(logits[0])[1]

def test(path):
  train_dir = pj(path, 'training')
  check_path = tf.train.latest_checkpoint(train_dir)
  print 'Restoring from:', check_path
  net = NetClf(check_path, gpu)
  data = ut.load(pj(path, 'test.pk'))

  labels = []
  probs = []
  accs = []
  for i in xrange(len(data)):
    ex = data[i]
    label = ex['is_gripping']
    ex = {k : ig.uncompress(ex[k]) for k in im_names}
    prob = net.predict(**ex)
    print prob, label
    pred = int(prob >= 0.5)

    labels.append(label)
    probs.append(prob)
    accs.append(pred == label)
  labels = np.array(labels, 'bool')
  probs = np.array(probs, 'float32')
  accs = np.array(accs)

  print 'Accuracy:', np.mean(accs)
  print 'mAP:', sklearn.metrics.average_precision_score(labels, probs)


def run(todo = 'all', 
        out_dir = '../results/grasp/grasp-v1', 
        restore = 0):

  todo = ut.make_todo(todo, 'im train test')

  if 'im' in todo:
    write_data(out_dir)

  if 'train' in todo:
    train(out_dir, restore = restore)

  if 'test' in todo:
    test(out_dir)
