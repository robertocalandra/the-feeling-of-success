import os, sys, img as ig, util as ut, scipy.io, numpy as np

#SiftPath = '../lib/vlfeat_mac/mmmikael-vlfeat-1323904/bin/maci/sift'
SiftPath = '/csail/vision-billf5/aho/mono/lib/vlfeat-0.9.14/bin/glnxa64/sift'

#from _vlfeat import *
def vl_sift(im, frames = None, orientations = False, peak_thresh = None, edge_thresh = None):
  """ Compute SIFT keypoints and descriptors using VLFeat binary.
  Should be thread-safe. """
  ut.check(frames is None or frames.shape[1] == 4)
  # frame_fname = '../tmp/vl_frames.frame'
  # im_fname1 = '../tmp/vl_im.png'
  # im_fname2 = '../tmp/vl_im.pgm'  
  # out_fname = '../tmp/vl_out.sift'
  frame_fname = ut.make_temp('.frame')
  im_fname1 = ut.make_temp('.png')
  im_fname2 = ut.make_temp('.pgm')
  out_fname = ut.make_temp('.sift')
  #ut.write_lines(frame_fname, ('%f %f %f 0 0 %f' % (pt[0], pt[1], s, s) for pt in pts for s in scales))
  ig.save(im_fname1, im)
  os.system('convert %s %s' % (im_fname1, im_fname2))
  frame_opt = ''
  if frames is not None:
    ut.write_lines(frame_fname, ('%f %f %f %f' % tuple(f) for f in frames))
    frame_opt = '--read-frames %s' % frame_fname
  orientation_opt = '--orientations' if orientations else ''
  peak_opt = '--peak-thresh %f' % peak_thresh if peak_thresh is not None else ''
  edge_opt = '--edge-thresh %f' % edge_thresh if edge_thresh is not None else ''
  ut.sys_check("%s %s %s %s -o %s %s %s" % (SiftPath, im_fname2, frame_opt, orientation_opt, out_fname, peak_opt, edge_opt))
  sift = read_sift(out_fname)
  os.system('rm %s %s %s' % (im_fname1, im_fname2, out_fname))
  return sift

def read_sift(sift_fname):
  """ Feature format: [[x, y, scale, orientation], ...] """
  lines = ut.lines(sift_fname)
  if len(lines):
    fd = np.array([map(float, line.split()) for line in lines])
    f = fd[:,:4]
    d = np.uint8(fd[:,4:])
    return f, d
  else:
    return np.zeros((4, 0)), np.uint8(np.zeros((128, 0)))
  

