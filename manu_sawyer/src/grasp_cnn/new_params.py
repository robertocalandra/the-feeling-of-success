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
    dsdir = '../results/dset/v1-split%d' % split_num,
    base_lr = 1e-4,
    #base_lr = 1e-2,
    #base_lr = 0.01,
    #base_lr = 1e-3,
    lr_gamma = 0.5,
    #step_size = 1500,
    step_size = 2000,
    batch_size = 16,
    opt_method = 'adam',
    #opt_method = 'momentum',
    cls_style = 'mlp-1',
    #model_iter = 5000,
    model_iter = 7000,
    train_iters = 5001)

def gel_im_v1(n):
  pr = base_v1(n).updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/gel-im-v1-split-%d' % n,
    inputs = ['gel', 'im'])
  return pr

def gel_v1(n):
  pr = base_v1(n).updated(
    description = 'GelSight',
    resdir = '../results/grasp-params/gel-v1-split-%d' % n,
    inputs = ['gel'])
  return pr

def im_v1(n):
  pr = base_v1(n).updated(
    description = 'images',
    resdir = '../results/grasp-params/im-v1-split-%d' % n,
    inputs = ['im'])
  return pr

def depth_v1(n):
  pr = base_v1(n).updated(
    description = 'images',
    resdir = '../results/grasp-params/depth-v1-split-%d' % n,
    inputs = ['depth'])
  return pr

def press_v1(n):
  pr = base_v1(n).updated(
    description = 'GelSight + images',
    resdir = '../results/grasp-params/press-split-%d' % n,
    inputs = ['press'])
  return pr
