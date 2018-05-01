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

# def base_v1():
#   return Params(
#     #dsdir = '/media/4tb/owens/grasp-results/dset-v2/v2-full',
#     dsdir = '/z/owens/v2-full',
#     base_lr = 1e-4,
#     lr_gamma = 0.5,
#     step_size = 2000,
#     batch_size = 16,
#     opt_method = 'adam',
#     cls_style = 'mlp-1',
#     model_iter = 7000,
#     train_iters = 7001)



# def base_v1():
#   return Params(
#     #dsdir = '/media/4tb/owens/grasp-results/dset-v2/v2-full',
#     dsdir = '/z/owens/v2-full',
#     base_lr = 1e-4,
#     lr_gamma = 0.1,
#     step_size = 4000,
#     batch_size = 16,
#     opt_method = 'adam',
#     cls_style = 'mlp-1',
#     model_iter = 9000,
#     train_iters = 9001)


def base_v1():
  epochs = 20
  dset_batches = 9296/16
  return Params(
    #dsdir = '/media/4tb/owens/grasp-results/dset-v2/v2-full',
    dsdir = '/z/owens/v2-full',
    base_lr = 1e-4,
    lr_gamma = 0.1,
    #step_size = 4000,
    step_size = 10*dset_batches,
    batch_size = 16,
    opt_method = 'adam',
    cls_style = 'mlp-1',
    model_iter = epochs*dset_batches,
    train_iters = epochs*dset_batches+1)

# def gel_im_v1():
#   pr = base_v1().updated(
#       description = 'GelSight + images',
#       resdir = '/media/4tb/owens/grasp-results/final/gel-im-v1',
#       inputs = ['gel', 'im'])
#   return pr


# def im_v1():
#   pr = base_v1().updated(
#       description = 'Images only',
#       resdir = '/z/owens/grasp-results/final/im-v1',
#       inputs = ['im'])
#   return pr

# def gel_im_v1():
#   pr = base_v1().updated(
#       description = 'GelSight + images',
#       resdir = '/media/4tb/owens/grasp-results/final/gel-im-v1',
#       inputs = ['gel', 'im'])
#   return pr


def gel_im_v1():
  pr = base_v1().updated(
      description = 'Images only',
      resdir = '/z/owens/grasp-results/final/gel-im-v1',
      inputs = ['im', 'gel'])
  return pr


def im_v1():
  pr = base_v1().updated(
      description = 'Images only',
      resdir = '/z/owens/grasp-results/final/im-v1',
      inputs = ['im'])
  return pr
