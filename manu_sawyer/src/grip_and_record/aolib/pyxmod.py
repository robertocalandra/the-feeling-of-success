import os, imp, sys, traceback, tempfile

# usage:
# import pyxmod
# foo = pyxmod.load('foo', ['bar.o'])
#
# Compiles foo.pyx and links it with the (optional) object file bar.o.
# Returns the resulting module as foo.

# if COMPILE is false, then load() becomes a normal import

def load(name, deps = [], rebuild = True):
  if rebuild:
    return loader.load(name, deps)
  else:
    return __import__(name)

def print_error():
  """ IPython does not seem to print runtime errors from load()ed
  modules correctly, as line numbers from .pyx files are missing.
  This function should print them correctly. """
  traceback.print_tb(sys.last_traceback)

class Loader:
  def __init__(self):
    self.modules = {}
    
  def load(self, name, deps = []):
    reload = True
    # only reload if the timestamp has changed
    if name in self.modules:
      edit_time = max([os.stat('%s.pyx' % name).st_mtime] + [os.stat(dep).st_mtime for dep in deps])
      if not hasattr(self.modules[name], '__file__')  or os.stat(self.modules[name].__file__).st_mtime > edit_time:
        reload = False
    if reload:
      self.modules[name] = compile_and_load(name, deps)
    return self.modules[name]

# a signal to areload.py to not reload this file
__no_autoreload__ = True

def compile_and_load(name, deps = []):
  """ Build and load the module corresponding to [name].pyx.  deps is
  the list of object files to be linked against. """
  # leaks a directory in /tmp; should fix
  work_dir = tempfile.mkdtemp()
  new_name = make_temp('.so', dir = work_dir).replace('.so', '')
  os.system('cp %s.pyx %s.pyx' % (name, new_name))
  if len(deps) > 0:
    os.system('cp %s %s' % (' '.join(deps), work_dir))
  obj_files = ' '.join(os.path.join(work_dir, d) for d in deps if d.endswith('o'))
  
  # cmd = ('/afs/csail/group/vision/roger/epd/bin/cython %(new_name)s.pyx;'
  #        ' g++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -L'
  #        ' /afs/csail/group/vision/roger/epd/lib -lpython2.7'
  #        ' -I /afs/csail/group/vision/roger/epd/include/python2.7'
  #        ' -I /afs/csail/group/vision/roger/epd/lib/python2.7/site-packages/numpy/core/include'
  #        ' -o %(new_name)s.so %(new_name)s.c %(obj_files)s') % locals()
  py_dir = '/data/vision/billf/aho-billf/conda/'
  py_inc = os.path.join(py_dir, 'include/python2.7')
  py_lib = os.path.join(py_dir, 'lib')
  np_inc = os.path.join(py_dir, 'lib/python2.7/site-packages/numpy/core/include')
  cython_cmd = 'cython'
  
  cmd = ('%(cython_cmd)s %(new_name)s.pyx;'
         ' g++ -fopenmp -shared -pthread -fPIC -fwrapv -O2 -Wall '
         '`python-config --libs` -L`python-config --prefix`/lib `python-config --ldflags` `python-config --includes` -I `python -c "import numpy; print numpy.get_include()"`'
#         ' %(py_lib)s -lpython2.7'
#         ' -I %(py_inc)s'
#         ' -I /afs/csail/group/vision/roger/epd/lib/python2.7/site-packages/numpy/core/include'
         ' -o %(new_name)s.so %(new_name)s.c %(obj_files)s') % locals()
  print >>sys.stderr, cmd
  os.system(cmd)
  
  print >>sys.stderr, 'loading cython module', new_name + '.so'
  m = imp.load_dynamic(os.path.split(new_name)[1], new_name + '.so')
  return m

def make_temp(ext, dir):
  fd, fname = tempfile.mkstemp(ext, dir = dir)
  os.close(fd)
  return fname

# module-level to avoid reloading already-loaded modules
loader = Loader()
