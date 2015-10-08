import os, sys
import numpy as np
import pandas as pd

def read_vad(filename):
  vad = pd.read_csv(filename)
  return vad

def stem(fn):
  return os.path.splitext(os.path.basename(fn))[0]

def vad_for(fn, vad):
  fn_stem = stem(fn)
  return vad.ix[vad["f_id"]==fn_stem,("start","end")].values

def write_vad(vad, fn):
  np.savetxt(fn, vad, fmt="%d")

def output_vad_fn(fn, dir):
  return os.path.join(dir, stem(fn) + ".vad")

vad_all = read_vad(sys.argv[2])

for fn in sys.argv[4:]:
  vad = vad_for(fn, vad_all)
  if vad.shape[0] > 0:
    write_vad(vad, output_vad_fn(fn, sys.argv[1]) + sys.argv[3])

