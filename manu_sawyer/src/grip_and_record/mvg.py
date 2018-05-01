import aolib.util as ut
import itertools as itl
import numpy as np
import planefit
import pylab
import random
import scipy.linalg
import scipy.misc

na = np.newaxis

def F_from_Ps(P1, P2, cam2_center = None):
  if cam2_center is None:
    cam2_center = center_from_P(P2)
  e = np.dot(P1, ut.homog(cam2_center))
  F = np.dot(np.dot(np.linalg.pinv(P2.T), P1.T), cross_product_matrix(e))
  F = F/F[2,2]
  return F
  # http://www.csse.uwa.edu.au/~pk/research/matlabfns/Projective/fundfromcameras.m
  #C1 = 

def F_from_matches(f1, f2):
  from ext import cv2
  return cv2.findFundamentalMat(f1, f2, cv2.FM_RANSAC)[0]
  
def cross_product_matrix(v):
  """
  >>> C = cross_product_matrix(array([1.0, 1.0, 1.0]))
  >>> C
  array([[ 0., -1.,  1.],
    [ 1.,  0., -1.],
    [-1.,  1.,  0.]])
  >>> dot(C, [1.0, 2.0, 1.0])
  array([-1.,  0.,  1.])
  >>> dot(array([-1.,  0.,  1.]), array([1.0, 1.0, 1.0]))
  0.0
  >>> dot(array([-1.,  0.,  1.]), array([1.0, 2.0, 1.0]))
  0.0
  """
  return np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])

def cv_triangulate(Ps, pt_projs, im_shape = None):
  assert len(Ps) == 2
  from ext import cv2
  Xs = cv2.triangulatePoints(np.array(Ps[0], 'd'), np.array(Ps[1], 'd'), np.array(pt_projs[0], 'd').T, np.array(pt_projs[1], 'd').T)
  return ut.inhomog(Xs).T

def norm_homog(w, h):
  return np.array([[2.0 / w, 0.0, -1.0],
                   [0.0, 2.0 / h, -1.0],
                   [0.0, 0.0,      1.]])

def solve_0_system(A):
  """
  >>> solve_0_system(array([[1., 1., 0.], [2., 3., -1.], [10., 2., 8.]]))
  array([-0.57735027,  0.57735027,  0.57735027])
  """
  U, S, Vt = np.linalg.svd(A)
  return Vt[-1, :]

def precondition_Ps(Ps, projs, im_shapes):
  assert len(Ps) == len(projs) == len(im_shapes)
  assert all([len(projs[0]) == len(projs[i]) for i in xrange(len(projs))])
  new_Ps = []
  new_projs = []
  for P, xs, sz in itl.izip(Ps, projs, im_shapes):
    H = norm_homog(sz[1], sz[0])
    new_P = np.dot(H, P)
    new_Ps.append(new_P)
    new_projs.append([ut.inhomog(np.dot(H, ut.homog(pt))) for pt in xs])
  return new_Ps, new_projs

def triangulate_linear(Ps, projs, im_shapes):
  Ps, projs = precondition_Ps(Ps, projs, im_shapes)
  X = []
  for i in xrange(len(projs[0])):
    A = []
    for P, P_xs in itl.izip(Ps, projs):
      x = P_xs[i]
      A.append(-x[0]*P[2, :] + P[0, :])
      A.append(x[1]*P[2, :] - P[1, :])
    X.append(ut.inhomog(solve_0_system(np.vstack(A))))
  return np.array(X)

def triangulate_linear_pt(Ps, projs, im_shapes):
  return triangulate_linear(Ps, np.array([p[np.newaxis, :] for p in projs]), im_shapes)
  
def triangulate_nonlinear_many(Ps, projs, im_sizes):
  Xs = []
  for i in xrange(len(projs[0])):
    Xs.append(triangulate_nonlinear_pt(Ps, [p[i] for p in projs], im_sizes))
  return np.array(Xs)

# def triangulate_nonlinear_pt(Ps, projs, im_sizes):
#   [X] = triangulate_linear(Ps, [[p] for p in projs], im_sizes)
#   Ps, projs = precondition_Ps(Ps, [[p] for p in projs], im_sizes)
#   projs = [p[0] for p in projs]
  
#   _, _, T = np.linalg.svd(ut.homog(X)[np.newaxis, :]);
#   T = T[:, range(1, T.shape[1]) + [0]]
#   Qs = [np.dot(P, T) for P in Ps]

#   Y = np.zeros(3)
#   eprev = np.inf
#   for n in xrange(10):
#     e, J = resid(Y, projs, Qs)
#     if 1 - np.linalg.norm(e) / np.linalg.norm(eprev) < 1000*np.finfo(np.float).eps:
#       break
#     eprev = e
    
#     try:
#       Y = Y - np.linalg.solve(np.dot(J.T, J), -np.dot(J.T, e))
#     except np.linalg.linalg.LinAlgError as err:
#       if 'Singular matrix' in err.message:
#         break
#       else:
#         raise
#     #Y = Y - np.linalg.lstsq(np.dot(J.T, J), -np.dot(J.T, e))[0]
#   return ut.homog_transform(T, Y)

# def resid(Y, projs, Qs):
#   errs, J_blocks = [], []
#   for u, Q in itl.izip(projs, Qs):
#     q = Q[:, :3]
#     x0 = Q[:, 3]
#     x = np.dot(q, Y) + x0
#     errs.append(x[:2]/x[2] - u)
#     # I don't understand how this line was computed; just algebraically taking the derivative?
#     J = np.array([x[2]*q[0, :] - x[0]*q[2, :],
#                   x[2]*q[1, :] - x[1]*q[2, :]], 'd')/x[2]**2
#     J_blocks.append(J)
#   return np.concatenate(errs), np.vstack(J_blocks)

def triangulate_nonlinear_pt(Ps, projs, im_sizes):
  [X] = triangulate_linear(Ps, [[p] for p in projs], im_sizes)
  Ps, projs = precondition_Ps(Ps, [[p] for p in projs], im_sizes)
  projs = np.array([p[0] for p in projs])
  def resid(X):
    errs = np.zeros(2*len(Ps))
    for i in xrange(len(Ps)):
      x = ut.homog_transform(Ps[i], X)
      errs[0 + 2*i] = x[0] - projs[i, 0]
      errs[1 + 2*i] = x[1] - projs[i, 1]
    return errs
      
  return scipy.optimize.leastsq(resid, X, maxfev = 30)[0]

def decompose_P(P):
  K, R = scipy.linalg.rq(P[:3, :3])
  d = K[2,2]
  K /= d
  if K[0, 0] < 0 or K[1,1] < 0:
    D = np.diag([np.sign(K[0,0]), np.sign(K[1,1]), 1.])
    K = np.dot(K, D)
    R = np.dot(D, R) if d > 0 else np.dot(D, -R)
    
  assert K[0, 0] > 0 and K[1,1] > 0
  # not strictly necessary, but should be true for all of my code
  # ... not true for bundler code
  # assert K[0, 2] >= 0  and K[1, 2] >= 0
  # assert np.allclose(np.dot(K, R), P[:3,:3])
                     
  t = np.linalg.solve(K, P[:, 3])
  return (K, R, t)

def compose_P(K, R = np.eye(3), t = np.zeros(3)):
  return np.dot(K, np.hstack([R, t[:, np.newaxis]]))

def center_from_P(P):
  return ut.inhomog(solve_0_system(P))

def reproj_error(P, X, x):
  d = ut.homog_transform(P, X) - x
  return np.dot(d, d)

def proj_dist(P, X, x):
  return pylab.dist(ut.homog_transform(P, X), x)

def pixel_ray_matrix(R, K):
  """ Does not account for camera translation.  Just the viewing direction """
  return np.dot(R.T, np.linalg.inv(K))

def ray_directions(K, im_shape, R = np.eye(3), normalize = True):
  h, w = im_shape[:2]
  y, x = np.mgrid[:h, :w]
  y = np.single(y)
  x = np.single(x)

  # rays = np.zeros((h, w, 3))
  # rays[:,:,0] = (x-K[0,2])/K[0, 0]
  # rays[:,:,1] = (y-K[1,2])/K[1, 1]
  # rays[:,:,2] = 1
  
  rays = np.dot(np.dot(R.T, np.linalg.inv(K)), np.array([x.flatten(), y.flatten(), np.ones(x.size)]))
  rays = rays.reshape((3, h, w)).transpose([1, 2, 0])
  assert np.allclose(rays[20, 30], np.dot(np.dot(R.T, np.linalg.inv(K)), np.array([30, 20, 1.])))
  
  if normalize:
    rays = ut.normalize_im(rays)

  return rays

def pixel_ray(P, x, y):
  K, R, t = decompose_P(P)
  return np.dot(R.T, ut.normalized(np.array([(x-K[0,2])/K[0, 0],
                                             (y-K[1,2])/K[1, 1],
                                             1.0])))
  
def brute_force_triangulate(Ps, pt_projs, min_dist, max_dist, sample_step = 0.01):
  assert len(Ps) == 2
  pt_projs = map(np.asarray, pt_projs)
  
  P1, P2 = Ps

  K1, R1, t1 = decompose_P(P1)
  c1 = center_from_P(P1)
  D1 = pixel_ray_matrix(P1)
  
  vd1 = np.dot(D1, ut.homog(np.array(pt_projs[0]).T))
  vd1 = vd1 / np.sqrt(np.sum(vd1**2, axis = 0))

  best_err = np.inf + np.zeros(pt_projs[0].shape[0])
  best_dist = np.inf + np.zeros(pt_projs[0].shape[0])
  dists = np.arange(min_dist, max_dist, sample_step)
  for d in dists:
    X = c1[:, np.newaxis] + d * vd1
    proj = ut.homog_transform(P2, X)
    err = np.sum((proj.T - pt_projs[1])**2, axis = 1)
    good = (err < best_err)
    best_err[good] = err[good]
    best_dist[good] = d

  return c1 + (best_dist * vd1).T

def sample_n(nchoices, k, nresults, ordered = True):
  #print 'sample_n'
  seen = set()
  choices = range(nchoices)
  nck = scipy.misc.comb(nchoices, k)
  if not ordered and nck < 10e6 or nck <= nresults:
    opts = list(itl.combinations(range(nchoices), k))
    return ut.sample_at_most(opts, nresults)
  
  res = []
  while len(seen) < nresults:
    inds = tuple(random.sample(choices, k))
    if not ordered:
      inds = tuple(sorted(inds))
    if inds not in seen:
      seen.add(inds)
      res.append(np.array(inds))
      
  return res
  
def ransac_sample(nchoices, nsample, niters):
  seen = set()
  choices = range(nchoices)
  for i in xrange(niters):
    inds = tuple(random.sample(choices, nsample))
    if inds not in seen:
      seen.add(inds)
      yield np.array(inds)

def ransac_sample_repeat(nchoices, nsample, niters):
  choices = range(nchoices)
  for i in xrange(niters):
    yield tuple(random.sample(choices, nsample))

def greedy_choose_mindist(ts, inds, min_dist):
  import scipy.spatial.distance
  dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts, 'euclidean'))
  chosen = []
  for ii in ut.shuffled(range(len(inds))):
    if np.all(dists[inds[ii], chosen] >= min_dist):
      chosen.append(inds[ii])

  if 1 and len(chosen) > 1:
    a, b = random.sample(chosen, 2)
    assert (pylab.dist(ts[a], ts[b]) >= min_dist)
  return chosen

def greedy_choose_minangle(Ps, X, ordered_inds, min_angle):
  chosen = []
  chosen_rays = []
  for i in ordered_inds:
    ray_i = -X + center_from_P(Ps[i])
    for ray_j in chosen_rays:
      if ut.angle_between(ray_i, ray_j) <= min_angle:
        break
    else:
      chosen.append(i)
      chosen_rays.append(ray_i)
  return chosen

def without(xs, ys):
  s = set(ys)
  return [x for x in xs if x not in s]

def triangulate_ransac_pt(Ps, projs, im_shapes, ransac_thresh, max_pt_dist = 5.):
  best_inliers = []
  CHOSEN_MIN_ANGLE = True
  MIN_ANGLE = np.radians(1)
  MAX_ANGLE = np.radians(120)

  INLIER_MINDIST = False
  INLIER_MINANGLE = True
  
  MIN_TRI_DIST = False
  TRI_DIST = 1.
  
  ts = np.array([center_from_P(P) for P in Ps])
  for inds in ransac_sample(len(Ps), 2, 300):
    Ps_s, projs_s, shapes_s = ut.take_inds_each([Ps, projs, im_shapes], inds)
    X = triangulate_nonlinear_pt(Ps_s, projs_s, shapes_s)

    if CHOSEN_MIN_ANGLE:
      angle = ut.angle_between(-X + ts[inds[0]], -X + ts[inds[1]])
      if angle <= MIN_ANGLE or angle >= MAX_ANGLE:
        continue
    
    if MIN_TRI_DIST and pylab.dist(ts[inds[0]], ts[inds[1]]) <= TRI_DIST:
      continue
    
    inliers = []
    for pi, (P, proj) in enumerate(zip(Ps, projs)):
      if reproj_error(P, X, proj) <= ransac_thresh \
             and in_front_of(P, X) and pylab.dist(center_from_P(P), X) <= max_pt_dist:
        inliers.append(pi)
    inliers = np.array(inliers)
    
    #[inliers] = np.nonzero(np.array([reproj_error(P, X, proj) <= ransac_thresh for P, proj in zip(Ps, projs)]))
    if INLIER_MINDIST:
      inliers = greedy_choose_mindist(ts, inliers, 0.5)
      #inliers = greedy_choose_minangle(ts, inliers, 0.5)
      if len(inliers) < 2:
        continue
      
    if INLIER_MINANGLE:
      ordered_cams = [i for i in inds if i in inliers] + ut.shuffled(without(inliers, inds))
      inliers = greedy_choose_minangle(Ps, X, ordered_cams, MIN_ANGLE)
    
    if len(inliers) > len(best_inliers):
      best_inliers = inliers

  if len(best_inliers) == 0:
    final = [0, 1]
  else:
    final = best_inliers
    
  Ps_s, projs_s, shapes_s = ut.take_inds_each([Ps, projs, im_shapes], final)
  X = triangulate_nonlinear_pt(Ps_s, projs_s, shapes_s)
  print 'num inliers', len(best_inliers), 'of', len(Ps), 'mean reproj error', np.mean([reproj_error(P, X, x) for P, x in zip(Ps_s, projs_s)])
  return X, best_inliers

def triangulate_ransac_many(Ps, projs, im_shapes, ransac_thresh):
  Xs = []
  for i in xrange(len(projs[0])):
    Xs.append(triangulate_ransac_pt(Ps, [p[i] for p in projs], im_shapes, ransac_thresh))
  #return np.array(Xs)
  return Xs

  # for inds in ransac_sample(len(Ps), 

def vdir(P):
  K, R, t = decompose_P(P)
  return R[2, :]

def in_front_of(P, X):
  if np.ndim(X) == 1:
    return np.dot(vdir(P), X - center_from_P(P)) >= 0
  else:
    assert np.ndim(X) == 2
    dots = np.dot(vdir(P)[np.newaxis, :], (X - center_from_P(P)).T) >= 0
    assert dots.shape[0] == 1
    return dots.flatten()

def triangulate_ransac_pt2(Ps, projs, im_shapes):
  ts = np.array([center_from_P(P) for P in Ps])
  for inds in ransac_sample(len(Ps), 2, 300):
    Ps_s, projs_s, shapes_s = ut.take_inds_each([Ps, projs, im_shapes], inds)
    X = triangulate_nonlinear_pt(Ps_s, projs_s, shapes_s)

    if CHOSEN_MIN_ANGLE:
      angle = ut.angle_between(-X + ts[inds[0]], -X + ts[inds[1]])
      if angle <= MIN_ANGLE or angle >= MAX_ANGLE:
        continue
    
    if MIN_TRI_DIST and pylab.dist(ts[inds[0]], ts[inds[1]]) <= TRI_DIST:
      continue
    
    inliers = []
    for pi, (P, proj) in enumerate(zip(Ps, projs)):
      if reproj_error(P, X, proj) <= ransac_thresh \
             and in_front_of(P, X) \
             and pylab.dist(ts[pi], X) <= max_pt_dist:
        inliers.append(pi)
    inliers = np.array(inliers)

    if not set.issubset(set(inds), set(inliers)):
      continue
    
    if INLIER_MINDIST:
      inliers = greedy_choose_mindist(ts, inliers, 0.5)
      if len(inliers) < 2:
        continue
      
    if INLIER_MINANGLE:
      ordered_cams = list(inds) + ut.shuffled(without(inliers, inds))
      inliers = greedy_choose_minangle(Ps, X, ordered_cams, MIN_ANGLE)
      if len(inliers) < 2:
        continue
    
    if len(inliers) > len(best_inliers):
      best_inliers = inliers

  if len(best_inliers) == 0:
    final = [0, 1]
  else:
    final = best_inliers
    
  Ps_s, projs_s, shapes_s = ut.take_inds_each([Ps, projs, im_shapes], final)
  X = triangulate_nonlinear_pt(Ps_s, projs_s, shapes_s)
  print 'num inliers', len(best_inliers), 'of', len(Ps), 'mean reproj error', np.mean([reproj_error(P, X, x) for P, x in zip(Ps_s, projs_s)])
  return X, best_inliers

def triangulate_search(Ps, projs, im_shapes):
  projs = [p[0] for p in projs]
  ts = np.array([center_from_P(P) for P in Ps])
  (max_angle, X_best, tri) = (None, None, ut.Struct(inliers = []))
  for inds in ransac_sample(len(Ps), 2, 300):
    Ps_s, projs_s, shapes_s = ut.take_inds_each([Ps, projs, im_shapes], inds)

    # # does not seem to help
    # F = F_from_Ps(Ps_s[0], Ps_s[1], center_from_P(Ps_s[1]))
    # if not F_test_matches(F, projs_s[0][na, :], projs_s[1][na, :])[0]:
    #   continue
    
    X = triangulate_nonlinear_pt(Ps_s, projs_s, shapes_s)

    angle = ut.angle_between(-X + ts[inds[0]], -X + ts[inds[1]])
    pt_ok = (in_front_of(Ps_s[0], X) and in_front_of(Ps_s[1], X)) \
            and (max(reproj_error(P, X, x) for P, x in zip(Ps_s, projs_s)) <= 100)
    
    #pt_ok = (in_front_of(Ps_s[0], X) and in_front_of(Ps_s[1], X))

    # F = F_from_Ps(Ps_s[0], Ps_s[1], center_from_P(Ps_s[1]))
    # if max(reproj_error(P, X, x) for P, x in zip(Ps_s, projs_s)) > 100 and F_test_matches(F, projs_s[0][na, :], projs_s[1][na, :])[0]:
    #   asdf
    
    if pt_ok:
      if max_angle <= angle:
        max_angle = angle
        X_best = X
        tri = ut.Struct(inliers = inds,
                        angle_deg = np.degrees(angle),
                        in_front = (in_front_of(Ps_s[0], X), in_front_of(Ps_s[1], X)),
                        cam_dist = pylab.dist(ts[inds[0]], ts[inds[1]]),
                        reproj_errors = [reproj_error(P, X, x) for P, x in zip(Ps_s, projs_s)])
                        
  return X_best, tri

# def F_test_matches(F, f1, f2, thresh = 0.5):
#   #err = np.dot(ut.homog(f1.T).T, np.dot(F, ut.homog(f2.T)))
#   f1 = ut.homog(f1.T).T
#   f2 = ut.homog(f2.T).T
#   err = [np.dot(f2[i], np.dot(F, f1[i])) for i in xrange(f1.shape[0])]
#   return np.abs(err) <= thresh


  # p = np.dot(F[:, :2], f1) + F[:, 2]
  # b = np.abs(np.dot(f2, p[:2]) + p[2]) <= thresh

  # if 0:
  #   b1 = F_test_matches(F, f1[np.newaxis, :], f2[np.newaxis, :]).flatten()[0]
  #   assert b1 == b
  
  # return b

def F_test_matches(F, f1, f2, thresh = 2.):
  lines = F.dot(ut.homog(f1.T)).T
  lines /= ut.normax_padzero(lines[:, :2], 1)[:, np.newaxis]
  ok = np.abs(np.sum(lines * ut.homog(f2.T).T, 1)) <= thresh
  #assert np.all(ok == np.array([F_test_match(F, x1, x2) for x1, x2 in zip(f1, f2)]))
  return ok
  
def F_test_match2(F, f1, f2, thresh = 2.):
  """ from PMVS """
  line = np.dot(F, ut.homog(f1))
  f = (line[0]**2 + line[1]**2)**0.5
  if f == 0:
    return 0.
  line /= f
  return np.abs(np.dot(line, ut.homog(f2))) <= thresh

F_test_match = F_test_match2

homog_transform = ut.homog_transform


def solve_H(pts1, pts2):
  pass
  

# def plane_H(n, d, K1, K2, R2, t2):
#   # broken
#   #H = K' (R - tnT/V/) K-\ (13.2)
#   # Returns H such that a pixel on the plane x1 corresponds to a pixel x2 in view P2
#   # as x2 = H*x1. See HZ p.327. Assumes P1 = K1[I | 0] and P2 = K2 [R2 | t2].
#   return np.dot(np.dot(K2, (R2 - np.outer(t2, n)/d)), np.linalg.inv(K1))

def test_plane_H():
  if 1:
    npts = 20
    n = ut.normalized(np.array([1., 1, -1]))
    d = 2.
    X = pylab.randn(3, npts)
    # X[:2,:]*n[:2] + d + n[2]*X[2] = 0 ==> X[2] = (-d - X[:2,:]*n[:2])/n[2]
    X[2] = (-d - np.sum(X[:2]*n[:2,np.newaxis], axis = 0))/n[2]
    assert np.all(np.abs(np.dot(X.T, n) + d) <= 0.001) 

    K1 = np.eye(3)
    P1 = compose_P(K1, np.eye(3), np.zeros(3))
    x1 = ut.homog_transform(P1, X)

    K2 = 2*np.eye(3)
    R2 = np.eye(3)
    t2 = np.array([0.5, 0, 0])
    P2 = compose_P(K2, R2, t2)
    x2 = ut.homog_transform(P2, X)

    H = plane_H(n, d, K1, K2, R2, t2)
    x2_est = ut.homog_transform(H, x1)

    assert ut.a_eq(x2, x2_est)

  if 1:
    n = np.array([-0.09576725, -0.02749329, -0.995024  ])
    d = 12.842613230422947
    X = pylab.randn(3, npts)
    X[2] = (-d - np.sum(X[:2]*n[:2,np.newaxis], axis = 0))/n[2]
    # P1
    K1 = np.array([[ 184.46153519,    0.        ,  320.5       ],
           [   0.        , -184.46153519,  240.5       ],
           [   0.        ,    0.        ,    1.        ]])
    P1 = compose_P(K1, np.eye(3), np.zeros(3))
    x1 = ut.homog_transform(P1, X)
    # P2
    K2 = np.array([[ 184.46153519,    0.        ,  320.5       ],
           [   0.        , -184.46153519,  240.5       ],
           [   0.        ,    0.        ,    1.        ]])
    R2 = np.array([[ 0.99540027, -0.00263395,  0.09576725],
           [ 0.        ,  0.99962199,  0.02749329],
           [-0.09580347, -0.02736683,  0.995024  ]])
    t2 = np.array([-3.42297712,  6.86145016, -1.94439297])
    P2 = compose_P(K2, R2, t2)
    x2 = ut.homog_transform(P2, X)

    H = plane_H(n, d, K1, K2, R2, t2)
    x2_est = ut.homog_transform(H, x1)
    assert ut.a_eq(x2, x2_est)

# def fit_H(pts1, pts2):
#   # h1*w = u
#   # h2*w = v
#   A = []
#   for i in xrange(len(pts1)):
#     r3 = np.dot(pts[i]

  
# def test_fit_H():
  
  
def find_H_bruteforce(P1, P2, plane):
  pts = planefit.sample_plane(plane, 100, 1., noise = 0)
  proj1 = ut.homog_transform(P1, pts.T).T
  proj2 = ut.homog_transform(P2, pts.T).T
  H, ins = cv2.findHomography(proj1, proj2)
  assert np.all(ins == 1)
  assert np.allclose(ut.homog_transform(H, proj1.T).T, proj2)
  return H
