import os, sys
import numpy as np
import pandas as pd

def make_output_dir(dirname):
  try:
    os.makedirs(dirname)
  except OSError:
    pass

def read_vad(filename):
  vad = pd.read_csv(filename)
  return vad

def stem(fn):
  return os.path.splitext(os.path.basename(fn))[0]

def vad_for(fn, vad):
  fn_stem = stem(fn)
  return vad.ix[vad["f_id"]==fn_stem,("start","end")].values

def only_vad(vad, feat):
  result = np.empty((0, feat.shape[1]))
  for i in range(vad.shape[0]):
    start = vad[i,0]
    end = vad[i,1]
    interval = feat[start:(end+1),:]
    result = np.append(result, interval, axis=0)
  return result

def write_npy(feat, fn):
  np.save(fn, feat)

def write_raw(feat, fn):
  feat.astype(np.float32).tofile(fn)

def output_npy_fn(fn, dir):
  return os.path.join(dir, stem(fn) + ".npy")

def output_raw_fn(fn, dir):
  return os.path.join(dir, stem(fn) + ".raw")

def output_vad_fn(fn, dir):
  return os.path.join(dir, stem(fn) + ".vad")


if __name__ == "__main__":
    make_output_dir(sys.argv[1])
    vad_all = read_vad(sys.argv[2])

    for fn in sys.argv[3:]:
      feat = np.load(fn)
      vad = vad_for(fn, vad_all)
      if vad.shape[0] > 0:
        feat_vad = only_vad(vad, feat)
        mean = np.mean(feat_vad, axis=0)
        std = np.std(feat_vad, axis=0)
        feat_z = (feat-mean)/std
        write_npy(feat_z, output_npy_fn(fn, sys.argv[1]))
        # write_raw(feat_z, output_raw_fn(fn, sys.argv[1]))
    
