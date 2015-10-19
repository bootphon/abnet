import joblib, glob, os
from joblib import Parallel, delayed
from itertools import izip
from functools import partial
from collections import defaultdict
import numpy as np
from multiprocessing import cpu_count
from dtw import DTW
from spectral import Spectral
from scipy.io import wavfile
import sys

FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use
# basedir = "/fhgfs/bootphon/scratch/gsynnaeve/BUCKEYE/buckeye_modified_split_devtest/"


def do_fbank(fname):
    srate, sound = wavfile.read(fname)
    fbanks = Spectral(nfilt=N_FBANKS,         # nb of filters in mel bank
                      alpha=0.97,             # pre-emphasis
                      do_dct=False,           # we do not want MFCCs
                      fs=srate,               # sampling rate
                      frate=FBANKS_RATE,      # frame rate
                      wlen=FBANKS_WINDOW,     # window length
                      nfft=512,               # length of dft
                      do_deltas=False,        # speed
                      do_deltasdeltas=False,  # acceleration
                      compression="log"
                 )
    fb = fbanks.transform(sound)
    return fb


if __name__ == "__main__":
    try:
        os.makedirs(sys.argv[1])
    except OSError:
        pass
    for fname in sys.argv[2:]:
        bname = os.path.basename(fname)
        with open(os.path.join(sys.argv[1], bname[:-3] + 'npy'), 'wb') as wf:
            np.save(wf, do_fbank(fname))


