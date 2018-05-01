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

def base_v1(split_num):
  return Params(
    dsdir = '/media/4tb/owens/grasp-results/dset/v1-split%d' % split_num,
    base_lr = 1e-4,
    lr_gamma = 0.5,
    step_size = 2000,
    batch_size = 16,
    opt_method = 'adam',
    cls_style = 'mlp-1',
    model_iter = 7000,
    train_iters = 7001)

def gel_im_v1(n):
  pr = base_v1(n).updated(
      description = 'GelSight + images',
      resdir = '/home/rail/manu/grasp-results/models/gel-im-v1-%d' % n,
      inputs = ['gel', 'im'])
  return pr

def im_v1(n):
  pr = base_v1(n).updated(
      description = 'Images',
      resdir = '/home/rail/manu/grasp-results/models/im-v1-%d' % n,
      inputs = ['im'])
  return pr

def im_ee_v1(n):
  pr = base_v1(n).updated(
      description = 'Images + End effector',
      resdir = '/home/rail/manu/grasp-results/models/im-ee-v1-%d' % n,
      inputs = ['im', 'ee'])
  return pr

def gel_v1(n):
  pr = base_v1(n).updated(
      description = 'Gel only',
      resdir = '/home/rail/manu/grasp-results/models/gel-v1-%d' % n,
      inputs = ['gel'])
  return pr

def gel0_v1(n):
  pr = base_v1(n).updated(
    description = 'Only gel 0',
    resdir = '/home/rail/manu/grasp-results/models/gel0-v1-%d' % n,
    inputs = ['gel'],
    gels = [0])
  return pr

def gel1_v1(n):
  pr = base_v1(n).updated(
    description = 'Only gel 1',
    resdir = '/home/rail/manu/grasp-results/models/gel1-v1-%d' % n,
    inputs = ['gel'],
    gels = [1])
  return pr

def press_v1(n):
  pr = base_v1(n).updated(
    description = 'Hand-coded features',
    resdir = '/home/rail/manu/grasp-results/models/press-v1-%d' % n,
    inputs = ['press'])
  return pr

def depth_v1(n):
  pr = base_v1(n).updated(
    description = 'Depth only',
    resdir = '/home/rail/manu/grasp-results/models/depth-v1-%d' % n,
    inputs = ['depth'])
  return pr

def gel_im_depth_v1(n):
  pr = base_v1(n).updated(
      description = 'Gel + image + depth',
      resdir = '/home/rail/manu/grasp-results/models/gel-im-depth-v1-%d' % n,
      inputs = ['gel', 'im', 'depth'])
  return pr
