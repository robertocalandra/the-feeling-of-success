import numpy as np, tensorflow as tf, aolib.util as ut, aolib.img as ig, os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import vgg, h5py
import sklearn.metrics

pj = ut.pjoin

class Params(ut.Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  @property
  def train_dir(self):
    return ut.mkdir(pj(self.resdir, 'training'))

  @property
  def summary_dir(self):
    return ut.mkdir(pj(self.resdir, 'summary'))

#base_lr = 1e-4
base_lr = 1e-5
batch_size = 32

def gel_v1():
  return Params(
    description = 'GelSight only',
    dsdir = '../results/grasp-dset/v1/',
    resdir = '../results/grasp-params/gel-v1/',
    base_lr = base_lr,
    lr_gamma = 0.5,
    step_size = 1000,
    batch_size = batch_size,
    train_iters = 5000,
    inputs = ['gel'])

# def im_v1():
#   return Params(
#     description = 'GelSight only',
#     dsdir = '../results/grasp-dset/v1/',
#     resdir = '../results/grasp-params/im-v1/',
#     base_lr = base_lr,
#     lr_gamma = 0.5,
#     step_size = 1000,
#     batch_size = batch_size,
#     train_iters = 5000,
#     inputs = ['im'])

# def gel_im_v1():
#   return Params(
#     description = 'GelSight only',
#     dsdir = '../results/grasp-dset/v1/',
#     resdir = '../results/grasp-params/gel-im-v1/',
#     base_lr = base_lr,
#     lr_gamma = 0.5,
#     step_size = 2*1000,
#     batch_size = 16,
#     train_iters = 2*5000,
#     inputs = ['gel', 'im'])

def base_v2():
  return Params(
    dsdir = '../results/grasp-dset/v2/',
    #base_lr = 1e-4,
    base_lr = 1e-5,
    lr_gamma = 0.5,
    step_size = 1500,
    batch_size = 32,
    train_iters = 7000)

def gel_v2():
  return base_v2().updated(
    description = 'GelSight only',
    resdir = '../results/grasp-params/gel-v2/',
    inputs = ['gel'])

def im_v2():
  return base_v2().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-v2/',
    inputs = ['im'])

def ee_v2():
  return base_v2().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/ee-v2/',
    inputs = ['ee'])

def im_ee_v2():
  return base_v2().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-ee-v2/',
    inputs = ['im', 'ee'])

def gel_im_v2():
  pr = base_v2().updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-v2/',
    inputs = ['gel', 'im'])
  pr.batch_size /= 2
  pr.step_size *= 2
  pr.train_iters *= 2

def depth_v2():
  return base_v2().updated(
    description = 'Depth only',
    resdir = '../results/grasp-params/depth-v2/',
    inputs = ['depth'])

def press_v2():
  return base_v2().updated(
    description = 'Press',
    resdir = '../results/grasp-params/press-v2/',
    inputs = ['press'])

def gel0_only_v2():
  return base_v2().updated(
    description = 'GelSight 0 only',
    resdir = '../results/grasp-params/gel0-only-v2/',
    inputs = ['gel'],
    gels = [0])

def gel1_only_v2():
  return base_v2().updated(
    description = 'GelSight 1 only',
    resdir = '../results/grasp-params/gel1-only-v2/',
    inputs = ['gel'],
    gels = [1])





def base_v3():
  return Params(
    dsdir = '../results/grasp-dset/v2/',
    opt_method = 'adam',
    base_lr = 1e-5,
    lr_gamma = 0.1,
    step_size = 1500,
    batch_size = 32,
    train_iters = 4000)
    #train_iters = 3000)

def gel_v3():
  return base_v3().updated(
    description = 'GelSight only',
    resdir = '../results/grasp-params/gel-v3/',
    inputs = ['gel'])

def im_v3():
  return base_v3().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-v3/',
    inputs = ['im'])

def ee_v3():
  return base_v3().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/ee-v3/',
    inputs = ['ee'])

def im_ee_v3():
  return base_v3().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-ee-v3/',
    inputs = ['im', 'ee'])

def gel_im_v3():
  pr = base_v3().updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-v3/',
    inputs = ['gel', 'im'])
  pr.batch_size /= 2
  pr.step_size *= 2
  pr.train_iters *= 2

def depth_v3():
  return base_v3().updated(
    description = 'Depth only',
    resdir = '../results/grasp-params/depth-v3/',
    inputs = ['depth'])

def press_v3():
  return base_v3().updated(
    description = 'Press',
    resdir = '../results/grasp-params/press-v3/',
    inputs = ['press'])

def gel0_only_v3():
  return base_v3().updated(
    description = 'GelSight 0 only',
    resdir = '../results/grasp-params/gel0-only-v3/',
    inputs = ['gel'],
    gels = [0])

def gel1_only_v3():
  return base_v3().updated(
    description = 'GelSight 1 only',
    resdir = '../results/grasp-params/gel1-only-v3/',
    inputs = ['gel'],
    gels = [1])



# def base_v4():
#   return Params(
#     dsdir = '../results/grasp-dset/v4/',
#     opt_method = 'adam',
#     base_lr = 1e-5,
#     lr_gamma = 0.1,
#     step_size = 1500,
#     batch_size = 32,
#     train_iters = 4000)
#     #train_iters = 3000)

def base_v4():
  return Params(
    dsdir = '../results/grasp-dset/v4/',
    base_lr = 1e-5,
    lr_gamma = 0.5,
    step_size = 1500,
    batch_size = 32,
    opt_method = 'adam',
    train_iters = 7000)

def gel_v4():
  return base_v4().updated(
    description = 'GelSight only',
    resdir = '../results/grasp-params/gel-v4/',
    inputs = ['gel'])

def im_v4():
  return base_v4().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-v4/',
    inputs = ['im'])

def ee_v4():
  return base_v4().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/ee-v4/',
    inputs = ['ee'])

def im_ee_v4():
  return base_v4().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-ee-v4/',
    inputs = ['im', 'ee'])

def gel_im_v4():
  pr = base_v4().updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-v4/',
    inputs = ['gel', 'im'])
  pr.batch_size /= 2
  pr.step_size *= 2
  pr.train_iters *= 2
  return pr

def depth_v4():
  return base_v4().updated(
    description = 'Depth only',
    resdir = '../results/grasp-params/depth-v4/',
    inputs = ['depth'])

def press_v4():
  return base_v4().updated(
    description = 'Press',
    resdir = '../results/grasp-params/press-v4/',
    inputs = ['press'])

def gel0_only_v4():
  return base_v4().updated(
    description = 'GelSight 0 only',
    resdir = '../results/grasp-params/gel0-only-v4/',
    inputs = ['gel'],
    gels = [0])

def gel1_only_v4():
  return base_v4().updated(
    description = 'GelSight 1 only',
    resdir = '../results/grasp-params/gel1-only-v4/',
    inputs = ['gel'],
    gels = [1])

####################################################################################
def base_v5():
  return Params(
    dsdir = '../results/grasp-dset/v5/',
    base_lr = 1e-5,
    lr_gamma = 0.5,
    step_size = 1500,
    batch_size = 32,
    opt_method = 'adam',
    model_iter = 5000,
    train_iters = 5001)

def gel_v5():
  return base_v5().updated(
    description = 'GelSight only',
    resdir = '../results/grasp-params/gel-v5/',
    inputs = ['gel'])

def im_v5():
  return base_v5().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-v5/',
    inputs = ['im'])

def ee_v5():
  return base_v5().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/ee-v5/',
    inputs = ['ee'])

def im_ee_v5():
  return base_v5().updated(
    description = 'Images only',
    resdir = '../results/grasp-params/im-ee-v5/',
    inputs = ['im', 'ee'])

def gel_im_v5():
  pr = base_v5().updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-v5/',
    inputs = ['gel', 'im'])
  pr.batch_size /= 2
  pr.step_size *= 2
  pr.train_iters *= 2
  return pr

def depth_v5():
  return base_v5().updated(
    description = 'Depth only',
    resdir = '../results/grasp-params/depth-v5/',
    inputs = ['depth'])

# def press_v5():
#   return base_v5().updated(
#     description = 'Press',
#     resdir = '../results/grasp-params/press-v5/',
#     inputs = ['press'])

def gel0_only_v5():
  return base_v5().updated(
    description = 'GelSight 0 only',
    resdir = '../results/grasp-params/gel0-only-v5/',
    inputs = ['gel'],
    gels = [0])

def gel1_only_v5():
  return base_v5().updated(
    description = 'GelSight 1 only',
    resdir = '../results/grasp-params/gel1-only-v5/',
    inputs = ['gel'],
    gels = [1])


# def gel_fulldata_v5():
#   return base_v5().updated(
#     description = 'GelSight only',
#     resdir = '../results/grasp-params/gel-fulldata-v5/',
#     #dset_names = ['train', 'test'],
#     dset_names = ['full_unbalanced'],
#     inputs = ['gel'])

def gel_im_fulldata_v5():
  return base_v5().updated(
    description = 'GelSight + image trained on full dataset',
    resdir = '../results/grasp-params/gel-im-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 6500,
    dset_names = ['full_unbalanced'],
    inputs = ['gel', 'im'])

def im_fulldata_v5():
  return base_v5().updated(
    description = 'Image trained on full dataset',
    resdir = '../results/grasp-params/im-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 6500,
    dset_names = ['full_unbalanced'],
    inputs = ['im'])


def gel_fulldata_v5():
  return base_v5().updated(
    description = 'Gel trained on full dataset',
    resdir = '../results/grasp-params/gel-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 6500,
    dset_names = ['full_unbalanced'],
    inputs = ['gel'])

def mean_diff_v5():
  return base_v5().updated(
    description = 'Mean diff',
    resdir = '../results/grasp-params/mean-diff-v5/',
    use_clf = True,
    inputs = ['mean-diff'])

def gel_im_same_v5():
  pr = base_v5().updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-same-v5/',
    batch_size = 28,
    inputs = ['gel', 'im'])
  return pr
