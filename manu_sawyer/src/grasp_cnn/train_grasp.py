import grasp_net, grasp_params as ps, aolib.util as ut, os
pj = ut.pjoin

#gpus = [1,2]
gpus = [0,1,2,3]
#gpus = [1]

# def all_params():
#   return [ps.gel_v2(),
#           ps.im_v2(),
#           #ps.depth_v2(),
#           #ps.press_v2(), 
#           #ps.ee_v2(),
#           #ps.im_ee_v2(),
#           ps.gel0_only_v2(),
#           ps.gel1_only_v2(),
#           ps.gel_im_v2(),]

# def all_params():
#   return [ps.gel_v3(),
#           ps.im_v3(),
#           ps.depth_v3(),
#           ps.press_v3(), 
#           ps.ee_v3(),
#           ps.im_ee_v3(),
#           ps.gel0_only_v3(),
#           ps.gel1_only_v3(),
#           ps.gel_im_v3(),]


# def all_params():
#   return [# ps.gel_v4(),
#           # ps.im_v4(),
#           # ps.depth_v4(),
#           # #ps.press_v4(), 
#           # ps.ee_v4(),
#           # ps.im_ee_v4(),
#           # ps.gel0_only_v4(),
#           # ps.gel1_only_v4(),
#           #ps.gel_im_v4(),]

def all_params():
  #return [ps.gel_v5(),]
          # ps.im_v5(),
          # ps.depth_v5(),
          # ps.ee_v5(),
          # ps.im_ee_v5(),
          # ps.gel0_only_v5(),
          # ps.gel1_only_v5(),
          # ps.gel_im_v5(),]

  return [#ps.gel_v5(),
          #ps.im_v5(),
          #ps.depth_v5(),
          #ps.ee_v5(),
          # ps.im_ee_v5(),
          # ps.gel0_only_v5(),
          ps.gel1_only_v5(),]
          #ps.gel_im_same_v5(),]
          #]
    #ps.gel_v5_single()]

def train_all(make_data = False):
  if make_data:
    #grasp_net.write_data(all_params()[0].dsdir)
    grasp_net.write_data(ps.base_v5().dsdir)
    #return
    #return

  prs = all_params()
  for pr in prs:
    print 'Running:', pr.resdir.split('/')[-1]
    grasp_net.train(pr, gpus)

def eval_all(run=True, gpu_num=None,test_on_train=False):
  print """\\begin{tabular}"""
  for pr in all_params():
    if run:
      if gpu_num is not None:
        grasp_net.test(pr, gpu_num, test_on_train = test_on_train)
      else:
        grasp_net.test(pr, None, test_on_train = test_on_train)
    out_file = pj(pr.resdir, 'eval_results.pk')
    if os.path.exists(out_file):
      r = ut.load(out_file)
      print '%s & %s%% & %s%% \\\\' % (pr.description, ut.f2(100*r['acc']), ut.f2(100*r['ap']))
  print """\\end{tabular}"""    
