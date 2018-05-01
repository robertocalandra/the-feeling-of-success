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

def base_v2():
  return Params(
    dsdir = '/media/4tb/owens/grasp-results/dset/v7-full',
    base_lr = 1e-4,
    lr_gamma = 0.5,
    step_size = 2500,
    batch_size = 16,
    opt_method = 'adam',
    cls_style = 'mlp-1',
    model_iter = 7000,
    train_iters = 7001)

def gel_im_v2():
  pr = base_v2().updated(
    description = 'GelSight + images',
    resdir = ut.mkdir('../results/grasp-params/v2/gel-im-v2'),
    inputs = ['gel', 'im'])
  return pr

def im_v2():
  pr = base_v2().updated(
    description = 'images',
    resdir = '../results/grasp-params/v2/im-v2',
    inputs = ['im'])
  return pr

def gel_im_depth_v2():
  pr = base_v2().updated(
    description = 'GelSight + images + depth',
    resdir = ut.mkdir('../results/grasp-params/v2/gel-im-depth-v2'),
    inputs = ['gel', 'im', 'depth'])
  return pr

def depth_v2():
  pr = base_v2().updated(
    description = 'Depth only',
    resdir = ut.mkdir('../results/grasp-params/v2/depth-v2'),
    inputs = ['depth'])
  return pr

def im_vgg_legacy_v2():
  pr = base_v2().updated(
    description = 'images',
    cls_style = 'linear',
    resdir = '../results/grasp-params/v2/im-vgg-legacy-v2',
    inputs = ['im'])
  return pr
