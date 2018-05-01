import numpy as np, tensorflow as tf, aolib.util as ut, aolib.img as ig, os, random, h5py, sklearn.metrics
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import vgg
#vgg = nets.vgg
#import tensorflow.contrib.slim.nets as nets
#resnet_v1 = nets.resnet_v1
#import resnet, resnet_utils

pj = ut.pjoin

full_dim = 256
crop_dim = 224
train_iters = 10000
batch_size = 32
base_lr = 1e-3
#base_lr = 1e-3
#base_lr = 1e-2
gamma = 0.5
#step_size = 1000
step_size = 2500
#sample_dur_secs = 0.15
sample_dur_secs = 0.05
sample_fps = 60
gpu = '/gpu:0'
#init_path = '../results/resnet_v1_50.ckpt'
init_path = '../results/vgg_16.ckpt'
checkpoint_iters = 100
#finetune_top_only = True
finetune_top_only = False
model_style = 'diff'
augment = True
#augment = False
#model_style = 'dual'

def download_pretrained():
  # https://github.com/tensorflow/models/tree/master/slim
  ut.mkdir('../results')
  # ut.sys_check('wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz '
  #              ' -O ../results/resnet_v1_50_2016_08_28.tar.gz')
  # ut.sys_check('cd ../results; tar -xzf resnet_v1_50_2016_08_28.tar.gz')
  ut.sys_check('wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz '
               ' -O ../results/vgg_16_2016_08_28.tar.gz')
  ut.sys_check('cd ../results; tar -xzf vgg_16_2016_08_28.tar.gz')


# def extract_frames((vid_file, time, label, vid_idx, im_dir)):
#   examples = []
#   with ut.TmpDir() as tmp_dir:
#     # ut.sys_check('ffmpeg -i "%s" -vf scale=%d:%d -ss %f -t %f -r %d "%s/%%07d.png"' % \
#     #              (vid_file, full_dim, full_dim, time, 
#     #               sample_dur_secs, sample_fps, tmp_dir))
#     ut.sys_check('ffmpeg -ss %f -i "%s" -vf scale=%d:%d  -t %f -r %d "%s/%%07d.png"' % \
#                  (time, vid_file, full_dim, full_dim, 
#                   sample_dur_secs, sample_fps, tmp_dir))
#     for frame_idx, fname in enumerate(sorted(ut.glob(pj(tmp_dir, '*.png')))):
#       im_file = pj(im_dir, '%s_%05d_%d.png' % (vid_idx, frame_idx, label))
#       ut.sys_check('cp %s %s' % (fname, im_file))
#       examples.append((im_file, label))
#   return examples


# def extract_frames((vid_file, time, label, vid_idx, im_dir)):
#   examples = []
#   with ut.TmpDir() as tmp_dir:
#     ut.sys_check('ffmpeg -ss %f -i "%s" -vf scale=%d:%d  -t %f -r %d "%s/%%07d.png"' % \
#                  (time, vid_file, full_dim, full_dim, 
#                   0.05, sample_fps, tmp_dir))
#     fname = sorted(ut.glob(pj(tmp_dir, '*.png')))[0]
#     prev_file = pj(im_dir, 'prev_%s_%05d_%d.png' % (vid_idx, 0, label))
#     ut.sys_check('cp %s %s' % (fname, prev_file))
    
#   with ut.TmpDir() as tmp_dir:
#     # ut.sys_check('ffmpeg -i "%s" -vf scale=%d:%d -ss %f -t %f -r %d "%s/%%07d.png"' % \
#     #              (vid_file, full_dim, full_dim, time, 
#     #               sample_dur_secs, sample_fps, tmp_dir))
#     ut.sys_check('ffmpeg -ss %f -i "%s" -vf scale=%d:%d  -t %f -r %d "%s/%%07d.png"' % \
#                  (time, vid_file, full_dim, full_dim, 
#                   sample_dur_secs, sample_fps, tmp_dir))
#     for frame_idx, fname in enumerate(sorted(ut.glob(pj(tmp_dir, '*.png')))):
#       im_file = pj(im_dir, '%s_%05d_%d.png' % (vid_idx, frame_idx, label))
#       ut.sys_check('cp %s %s' % (fname, im_file))
#       examples.append((im_file, prev_file, label))
#   return examples


def extract_frames((vid_file, time, label, vid_idx, im_dir, prev_free_time)):
  examples = []
  with ut.TmpDir() as tmp_dir:
    free_dur = 0.1
    ut.sys_check('ffmpeg -loglevel warning -ss %f -i "%s" -vf scale=%d:%d  -t %f -r %d "%s/%%07d.png"' % \
                 (prev_free_time, vid_file, full_dim, full_dim, free_dur, sample_fps, tmp_dir))
    #fname = sorted(ut.glob(pj(tmp_dir, '*.png')))[0]
    assert len(ut.glob(pj(tmp_dir, '*.png'))), 'no frames for prev_free_time' 
    fname = random.choice(sorted(ut.glob(pj(tmp_dir, '*.png'))))
    prev_file = pj(im_dir, 'prev_%s_%05d_%d.png' % (vid_idx, 0, label))
    ut.sys_check('cp %s %s' % (fname, prev_file))
    
  with ut.TmpDir() as tmp_dir:
    # ut.sys_check('ffmpeg -i "%s" -vf scale=%d:%d -ss %f -t %f -r %d "%s/%%07d.png"' % \
    #              (vid_file, full_dim, full_dim, time, 
    #               sample_dur_secs, sample_fps, tmp_dir))
    ut.sys_check('ffmpeg -loglevel warning -ss %f -i "%s" -vf scale=%d:%d  -t %f -r %d "%s/%%07d.png"' % \
                 (time, vid_file, full_dim, full_dim, 
                  sample_dur_secs, sample_fps, tmp_dir))
    for frame_idx, fname in enumerate(sorted(ut.glob(pj(tmp_dir, '*.png')))):
      im_file = pj(im_dir, '%s_%05d_%d.png' % (vid_idx, frame_idx, label))
      ut.sys_check('cp %s %s' % (fname, im_file))
      examples.append((im_file, prev_file, label, vid_file))
  return examples

def examples_from_db((db_file, im_dir)):
  examples = []
  try:
    with h5py.File(db_file, 'r') as db:
      #print db.keys()
      sc = lambda x : ig.scale(x, (full_dim, full_dim))
      for x in ['A', 'B']:
        im_file = ut.make_temp('.png', dir = im_dir)
        prev_file = ut.make_temp('.png', dir = im_dir)
        ig.save(im_file, sc(db['GelSight%s_image_post_gripping' % x]))
        ig.save(prev_file, sc(db['GelSight%s_image_pre_gripping' % x]))
        if 'is_gripping' in db:
          label = int(np.array(db['is_gripping'])[0])
        elif 'Is gripping?' in db:
          label = int(np.array(db['Is gripping?'])[0])
        else:
          raise RuntimeError('No label!')
        examples.append((im_file, prev_file, label, db_file))
  except:
    print 'Failed to open:', db_file
  return examples
              
def write_data(vid_path, out_dir, train_frac = 0.75):
  im_dir = ut.mkdir(pj(out_dir, 'ims'))
  in_data = []
  meta_files = sorted(ut.glob(vid_path, 'train', '*.txt'))
  print 'meta files:'
  for x in meta_files:
    print x
  print
  for meta_idx, meta_file in enumerate(meta_files):
    last_prev_time = 0.
    vid_file = meta_file.replace('.txt', '.mp4')
    for clip_idx, ex in enumerate(ut.read_lines(meta_file)):
      prev_time = last_prev_time
      vid_idx = '%05d_%05d' % (meta_idx, clip_idx)
      print ex
      s, time = ex.split()
      time = float(time)
      if s == 'p':
        label = 1
      elif s == 'n':
        label = 0
        last_prev_time = time
      else:
        raise RuntimeError()
      in_data.append((vid_file, time, label, vid_idx, im_dir, prev_time))
  print 'Writing:', len(in_data), 'sequences'
  meta_examples = ut.flatten(ut.parmap(extract_frames, in_data))
  meta_examples = ut.shuffled_with_seed(meta_examples)

  # add manu examples
  db_files = sorted(ut.sys_with_stdout('find ../data/manu-press -name "*.hdf5"').split())
  db_files = ut.shuffled_with_seed(db_files)
  print 'Train fraction:', train_frac
  num_train = int(train_frac * len(db_files))
  db_train = db_files[:num_train]
  db_test = db_files[num_train:]
  train_db_examples = ut.flatten(ut.parmap(examples_from_db, [(x, im_dir) for x in db_train]))
  test_db_examples = ut.flatten(ut.parmap(examples_from_db, [(x, im_dir) for x in db_test]))
  print 'Number of db train examples:', len(train_db_examples)
  print 'Number of meta examples:', len(meta_examples)
  train_examples = ut.shuffled_with_seed(meta_examples + train_db_examples)
  ut.write_lines(pj(out_dir, 'train.csv'), ['%s,%s,%d,%s' % x for x in train_examples])

  test_examples = ut.shuffled_with_seed(test_db_examples)
  ut.write_lines(pj(out_dir, 'test.csv'), ['%s,%s,%d,%s' % x for x in test_examples])

def make_tf(path):
  tf_file = pj(path, 'train.tf')
  if os.path.exists(tf_file):
    os.remove(tf_file)
  writer = tf.python_io.TFRecordWriter(tf_file)
  lines = ut.shuffled_with_seed(ut.read_lines(pj(path, 'train.csv')))
  print 'Number of examples:', len(lines)
  for line in lines:
    fname, prev_fname, label, _ = line.split(',')
    label = int(label)
    s = ut.read_file(fname, binary = True)
    s_prev = ut.read_file(prev_fname, binary = True)
    feat = {'im': tf.train.Feature(bytes_list = tf.train.BytesList(value = [s])),
            'im_prev': tf.train.Feature(bytes_list = tf.train.BytesList(value = [s_prev])),
            'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))}
    ex = tf.train.Example(features = tf.train.Features(feature = feat))      
    writer.write(ex.SerializeToString())
  writer.close()

def read_example(rec_queue):
  reader = tf.TFRecordReader()
  k, s = reader.read(rec_queue)
  feats = {
    'im' : tf.FixedLenFeature([], dtype=tf.string),
    'im_prev' : tf.FixedLenFeature([], dtype=tf.string),
    'label' : tf.FixedLenFeature([], tf.int64)
    }
  example = tf.parse_single_example(s, features = feats)
  im = tf.image.decode_png(example['im'])
  im_prev = tf.image.decode_png(example['im_prev'])
  im.set_shape((full_dim, full_dim, 3))
  im_prev.set_shape((full_dim, full_dim, 3))
  if 0:
    #im = tf.random_crop(im, (crop_dim, crop_dim, 3))
    im = tf.image.resize_images(im, [crop_dim, crop_dim])
    im_prev = tf.image.resize_images(im_prev, [crop_dim, crop_dim])
  else:
    im_combo = tf.concat([im, im_prev], 2)
    im_combo = tf.random_crop(im_combo, (crop_dim, crop_dim, 6))
    if augment:
      im_combo = tf.image.random_flip_left_right(im_combo)
      im_combo = tf.image.random_flip_up_down(im_combo)
      # See https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py for
      # an example of more aggressive brightness/contrast augmentation
      # The loss stays very high if you do not convert to float first: I think this is because
      # many values in the GelSight are close to 0/255, so they get saturated
      im_combo = tf.cast(im_combo, tf.float32)
      im_combo = tf.image.random_brightness(im_combo, max_delta=20)
      im_combo = tf.image.random_contrast(im_combo, lower=0.9, upper=1.1)
      #im_combo = tf.Print(im_combo, [tf.reduce_max(im_combo)])

    im = im_combo[:, :, :3]
    im_prev = im_combo[:, :, 3:]

  label = example['label']

  return im, im_prev, label


def read_data(path):
  #queues = [tf.train.string_input_producer(tf_files)]
  #example_list =  [read_example(queue) for queue in queues]
  tf_files = [pj(path, 'train.tf')]
  queue = tf.train.string_input_producer(tf_files)
  return tf.train.shuffle_batch(
    read_example(queue), batch_size = batch_size,
    capacity = 2000, min_after_dequeue = 500)

def normalize_ims(im):
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  return -1. + (2./255) * im 

def shape(x, d = None):
  s = x.get_shape().as_list()
  return s if d is None else s[d]

# def make_model(ims, train):
#   #ims = tf.Print(ims, ['ims before =', ims])
#   ims = normalize_ims(ims)
#   with slim.arg_scope(vgg.vgg_arg_scope()):
#     logits, _ = vgg.vgg_16(ims, is_training = train, update_top_only = finetune_top_only, num_classes = 2)
#     print shape(logits)
#     return logits


def make_model(ims, ims_prev, train):
  #ims = tf.Print(ims, ['ims before =', ims])
  ims = normalize_ims(ims)
  ims_prev = normalize_ims(ims_prev)
  if model_style == 'diff':
    logits, _ = vgg.vgg_dual_16(
      ims - ims_prev, ims, is_training = train, 
      update_top_only = finetune_top_only, 
      num_classes = 2)
  elif model_style == 'dual':
    logits, _ = vgg.vgg_dual_16(
      ims, ims_prev, is_training = train, 
      update_top_only = finetune_top_only, 
      num_classes = 2)
  return logits
    
def moving_avg(name, x, vals = {}, avg_win_size = 100):
  ut.add_dict_list(vals, name, x)
  return np.mean(vals[name][-avg_win_size:])

def train(path, restore = False):
  config = tf.ConfigProto(allow_soft_placement = True)
  with tf.Graph().as_default(), tf.device(gpu), tf.Session(config = config) as sess:
    global_step = tf.get_variable('global_step', [], initializer = 
                              tf.constant_initializer(0), trainable = False)
    ims, ims_prev, labels = read_data(path)
    #tf.summary.image('im', ims)
    logits = make_model(ims, ims_prev, train = True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    #logits = tf.Print(logits, ['logits =', logits[0, :], labels[0]])
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss)
    eq = tf.equal(tf.argmax(logits, 1), labels)
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))
    tf.summary.scalar('acc', acc)

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
      tf.train.Saver().restore(sess, tf.train.latest_checkpoint(train_dir))
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
        print 'Iteration %d,' % step, 'lr = ', lr_val, 'loss:',  \
              moving_avg('loss', loss_val), 'acc:', moving_avg('acc', acc_val)
  
def run(todo = 'all', 
        vid_path = '/home/ao/Videos/Webcam',
        #out_dir = '../results/press-data-v7/', 
        #out_dir = '../results/press-data-v8/', 
        #out_dir = '../results/press-data-v9/', 
        #out_dir = '../results/press-data-v10/', 
        out_dir = '../results/press-data-v11/', 
        restore = 0,
        train_frac = 0.75):

  todo = ut.make_todo(todo, 'im tf train test')

  if 'im' in todo:
    print vid_path
    write_data(vid_path, out_dir, train_frac = train_frac)

  if 'tf' in todo:
    make_tf(out_dir)

  if 'train' in todo:
    train(out_dir, restore = restore)

  if 'test' in todo:
    test(out_dir)


# class NetClf:
#   def __init__(self, model_file, gpu = '/cpu:0'):
#     self.sess = None
#     #self.train_path = train_path
#     self.model_file = model_file
#     self.gpu = gpu

#   def __del__(self):
#     self.deinit()
      
#   def init(self):
#     if self.sess is None:
#       #self.model_file = tf.train.latest_checkpoint(self.train_path)
#       print 'Restoring:',self.model_file
#       with tf.device(self.gpu):
#         tf.reset_default_graph()
#         print self.gpu
#         tf.Graph().as_default()
#         #self.sess = tf.Session()
#         #self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
#         self.sess = tf.Session()
#         #self.im_input = tf.placeholder(tf.float32, (1, crop_dim, crop_dim, 3), name = 'im_input')
#         self.im_input = tf.placeholder(tf.uint8, (1, crop_dim, crop_dim, 3), name = 'im_input')
#         ims = tf.cast(self.im_input, tf.float32)
#         self.logits = make_model(ims, train = False)
#         # var_list = slim.get_variables_to_restore()
#         # print 'Restoring:'
#         # for x in var_list:
#         #   print x.name
#         #self.sess.run(tf.global_variables_initializer())
#         #tf.train.Saver(var_list).restore(self.sess, self.model_file)
#         tf.train.Saver().restore(self.sess, self.model_file)
#         tf.get_default_graph().finalize()
        
#   def deinit(self):
#     if self.sess is not None:
#       self.sess.close()
#       self.sess = None

#   def format_im(self, im):
#     # im = ig.scale(im, self.full_shape)
#     # h_off = (im.shape[0] - crop_dim) // 2
#     # w_off = (im.shape[1] - crop_dim) // 2
#     # im = im[h_off : h_off + crop_dim, w_off : w_off + crop_dim]
#     # return im
#     return ig.scale(im, (crop_dim, crop_dim), 1)#.astype('float32')
    
#   def predict(self, im):
#     self.init()
#     im = self.format_im(im)
#     #print 'mean =', im.mean((0,1))
#     ut.tic()
#     [logits] = self.sess.run([self.logits], feed_dict = {self.im_input : im[None]})
#     print 'logits =', logits
#     ut.toc()
#     return ut.softmax(logits[0])[1]


class NetClf:
  def __init__(self, model_file, gpu = '/cpu:0'):
    self.sess = None
    self.model_file = model_file
    self.gpu = gpu

  def __del__(self):
    self.deinit()
      
  def init(self):
    if self.sess is None:
      #self.model_file = tf.train.latest_checkpoint(self.train_path)
      print 'Restoring:',self.model_file
      with tf.device(self.gpu):
        tf.reset_default_graph()
        print self.gpu
        tf.Graph().as_default()
        #self.sess = tf.Session()
        #self.sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        self.sess = tf.Session()
        #self.im_input = tf.placeholder(tf.float32, (1, crop_dim, crop_dim, 3), name = 'im_input')
        self.im_input = tf.placeholder(tf.uint8, (1, crop_dim, crop_dim, 3), name = 'im_input')
        self.im_prev_input = tf.placeholder(tf.uint8, (1, crop_dim, crop_dim, 3), name = 'im_prev_input')
        #ims = tf.cast(self.im_input, tf.float32)
        self.logits = make_model(self.im_input, self.im_prev_input, train = False)
        # var_list = slim.get_variables_to_restore()
        # print 'Restoring:'
        # for x in var_list:
        #   print x.name
        #self.sess.run(tf.global_variables_initializer())
        #tf.train.Saver(var_list).restore(self.sess, self.model_file)
        tf.train.Saver().restore(self.sess, self.model_file)
        tf.get_default_graph().finalize()
        
  def deinit(self):
    if self.sess is not None:
      self.sess.close()
      self.sess = None

  def format_im(self, im):
    return ig.scale(im, (crop_dim, crop_dim), 1)#.astype('float32')
    
  def predict(self, im, im_prev):
    self.init()
    im = self.format_im(im)
    im_prev = self.format_im(im_prev)
    #ut.tic()
    [logits] = self.sess.run([self.logits], feed_dict = {self.im_input : im[None],
                                                         self.im_prev_input : im_prev[None]})
    #print 'logits =', logits
    #ut.toc()
    return ut.softmax(logits[0])[1]

# class NetClf:
#   #def __init__(self, train_path, gpu = '/cpu:0'):
#   def __init__(self, train_path, gpu = '/gpu:0'):
#     self.sess = None
#     self.train_path = train_path
#     self.gpu = gpu
#     self.num_crops = 10

#   def __del__(self):
#     self.deinit()
      
#   def init(self):
#     if self.sess is None:
#       self.model_file = tf.train.latest_checkpoint(self.train_path)
#       print 'Restoring:',self.model_file
#       with tf.device(self.gpu):
#         tf.reset_default_graph()
#         print self.gpu
#         tf.Graph().as_default()
#         self.sess = tf.Session()
#         self.im_input = tf.placeholder(tf.uint8, (self.num_crops, crop_dim, crop_dim, 3), name = 'im_input')
#         ims = tf.cast(self.im_input, tf.float32)
#         self.logits = make_model(ims, train = False)
#         tf.train.Saver().restore(self.sess, self.model_file)
#         tf.get_default_graph().finalize()
        
#   def deinit(self):
#     if self.sess is not None:
#       self.sess.close()
#       self.sess = None

#   def format_im(self, im):
#     # im = ig.scale(im, self.full_shape)
#     # h_off = (im.shape[0] - crop_dim) // 2
#     # w_off = (im.shape[1] - crop_dim) // 2
#     # im = im[h_off : h_off + crop_dim, w_off : w_off + crop_dim]
#     # return im
#     #return [ig.scale(im, (crop_dim, crop_dim), 1)]*10 #.astype('float32')
#     dim = crop_dim
#     dh = (im.shape[0] - dim)
#     crops = np.zeros((self.num_crops, dim, dim, 3), dtype = np.uint8)
#     crops[0] = ut.crop_center(im, dim)
#     i = 1
#     for y in np.linspace(0, dh, 3).astype('l'):
#       dw = (im.shape[1] - dim)
#       for x in np.linspace(0, dw, 3).astype('l'):
#         crops[i] = im[y : y + dim, x : x + dim]
#         i += 1
#     return np.array(crops, 'float32')
    
  # def predict(self, im):
  #   self.init()
  #   im = self.format_im(im)
  #   #print 'mean =', im.mean((0,1))
  #   ut.tic()
  #   [logits] = self.sess.run([self.logits], feed_dict = {self.im_input : im})
  #   print 'logits =', logits
  #   ut.toc()
  #   return np.mean(map(ut.softmax, logits), axis = 0)[1]

def test(path, match_str = None):
  train_dir = pj(path, 'training')
  check_path = tf.train.latest_checkpoint(train_dir)
  print 'Restoring from:', check_path
  net = NetClf(check_path, gpu)
  examples = []
  for line in ut.read_lines(pj(path, 'test.csv')):
    s = line.split(',')
    #print match_str, s[3]
    print s
    if (match_str is not None) and (match_str not in s[3]):
      print 'skipping'
      continue
    examples.append((s[0], s[1], int(s[2]), s[3]))
  print 'Testing on:', len(examples), 'examples'

  labels = []
  probs = []
  accs = []
  table = []
  for i, ex in enumerate(examples):
    im_after = ig.load(ex[0])
    im_prev = ig.load(ex[1])
    label = ex[2]
    prob = net.predict(im_after, im_prev)
    #print prob, label
    pred = int(prob >= 0.5)

    labels.append(label)
    probs.append(prob)
    accs.append(pred == label)
    
    if i < 50:
      color = '#00DD00' if pred == label else '#DD0000'
      row = [im_after, im_prev, ut.font_color_html('pred = %.3f' % prob, color), 'gt = %d' % label]
      table.append(row)

  labels = np.array(labels, 'bool')
  probs = np.array(probs, 'float32')
  accs = np.array(accs)

  print 'Accuracy:', np.mean(accs)
  print 'mAP:', sklearn.metrics.average_precision_score(labels, probs)
  ig.show(table)
