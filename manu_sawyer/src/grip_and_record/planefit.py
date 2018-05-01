# from https://github.com/andrewowens/camo/blob/master/src/planefit.py
import aolib.util as ut
import mvg
import numpy as np


def plane_repok(plane):
  assert np.allclose(np.linalg.norm(plane[:3]), 1)
  return plane

def fit_plane(pts):
  # A = ut.homog(pts.T).T
  # b = mvg.solve_0_system(A)
  # b /= np.linalg.norm(b[:3])
  # return b
  axes, _, mean_pt = ut.pca(pts)
  w = axes[2]
  b = -np.dot(w, mean_pt)
  assert np.allclose(np.linalg.norm(w), 1)
  return np.array([w[0], w[1], w[2], b], 'd')

def sample_plane(plane, n, width, noise = 0):
  plane_repok(plane)
  e1, e2, _ = np.eye(3)
  v1 = e1 if (plane[0] == 0 and plane[1] == 0) else ut.normalized(np.array([-plane[1], plane[0], 0], 'd'))
  v2 = ut.normalized(np.cross(plane[:3], v1))
  #print 'dot', np.dot(v1, plane[:3]),   np.dot(v2, plane[:3])
  #print 'sample', np.sum(np.dot(np.array([v1, v2]).T, np.random.uniform(-width/2, width/2, (2, n))).T * plane[:3], 1)
  center = -plane[3]*plane[:3]
  #print 'dot2', np.dot(center, plane[:3]) + plane[3], plane
  pts = np.dot(np.array([v1, v2]).T, np.random.uniform(-width/2, width/2, (2, n))).T + center
  #print 'ins', len(plane_inliers(plane, pts, 0.05))
  pts += np.random.randn(*pts.shape)*noise
  return pts

def plane_from_3(pts):
  assert pts.shape == (3, 3)
  w = ut.normalized(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
  return np.array(list(w) + [-np.dot(w, pts[0])])

def dist_to_plane(plane, pts, signed = False):
  d = plane[3] + np.dot(pts, plane[:3])
  if signed:
    return d
  else:
    return np.abs(d)

def dist_from_plane_ptm(ptm, plane):
  dist = np.abs(plane[3] + (ptm[:, :, :3] * plane[:3]).sum(axis = 2))
  dist[ptm[:, :, 3] == 0] = np.inf
  return dist

def project_onto_plane(plane, pts):
  d = dist_to_plane(plane, pts, signed = True)
  proj = pts - d[:, None]*plane[:3]
  #print 'should be 0:', planefit.dist_to_plane(plane, proj)
  return proj

def projection_transformation(plane):
  w, b = plane[:3], plane[3]
  A = (np.outer(w, w) + np.eye(3))
  b = -w*b
  return A, b

def test_transform():
  projection_transformation

def plane_inliers(plane, pts, inlier_max_dist):
  err = plane[3] + np.dot(pts, plane[:3])
  #print 'err', err
  return np.nonzero(np.abs(err) <= inlier_max_dist)[0]

def transform_plane(R, t, plane):
  return np.array(list(R.T.dot(plane[:3])) + [np.dot(plane[:3], t) + plane[3]])

def fit_plane_ransac(pts, dist_thresh, ransac_iters = 200, filter_func = lambda plane : True, seed = 0):
  with ut.constant_seed(seed):
    if len(pts) < 3:
      return np.zeros(4), []

    best_inliers = range(3)
    for inds in mvg.ransac_sample(len(pts), 3, ransac_iters):
      plane = plane_from_3(pts[inds])
      if not np.all(plane[:3] == 0) and filter_func(plane):
        err = plane[3] + np.dot(pts, plane[:3])
        ins = np.nonzero(np.abs(err) <= dist_thresh)[0]
        if len(ins) > len(best_inliers):
          best_inliers = ins
          #print 'plane before', plane, 'after', fit_plane(pts[best_inliers])

    #print len(best_inliers)
    return fit_plane(pts[best_inliers]), best_inliers

def test_plane_fitting():
  noise = 0.0
  plane1 = np.array([0., 1./2**0.5, 1/2**0.5, 1.])
  plane2 = np.array([1., 0., 0., 5.])
  pts1 = sample_plane(plane1, 100, 1, noise)
  pts2 = sample_plane(plane2, 20, 1, noise)
  pts = np.vstack([pts1, pts2])
  plane, _ = fit_plane_ransac(pts, 0.05)
  #plane = fit_plane(pts1)
  print plane, 'should be', np.array([0., 1., 1., 1.])
  true_ins = plane_inliers(plane, pts, 0.05)
  print 'ninliers', len(true_ins), 'should be', len(plane_inliers(plane1, pts, 0.05)), 'other hypothesis', len(plane_inliers(plane2, pts, 0.05))
  ut.toplevel_locals()
  
