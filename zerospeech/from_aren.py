import sys
from dtw import DTW
import numpy as np
import joblib
MAX_LENGTH_WORDS = 6     # in phones
MIN_LENGTH_WORDS = 6     # in phones
MIN_FRAMES = 5           # in speech frames
#FBANKS_TIME_STEP = 0.01  # in seconds
FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40 # number of filterbanks to use

#bdir = '/fhgfs/bootphon/scratch/gsynnaeve/zerospeech/english_wavs/'
#bdir = '/fhgfs/bootphon/scratch/roland/abnet/buckeye_mdeltas/'
#bdir = '/fhgfs/bootphon/scratch/edunbar/buckeye_fb/fb_vadnorm/'

fin = sys.argv[1]
fout = sys.argv[2]
bdir = sys.argv[3]
dataset = sys.argv[4]


class Memoize:
    """Memoize(fn) 
    Will only work on functions with non-mutable arguments
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args):
        if not self.memo.has_key(args):
            self.memo[args] = self.fn(*args)
        return self.memo[args]


@Memoize
def do_fbank(fname):
    fn = bdir + fname + '.wav'
    try:
        try:
            with open(fn[:-3] + 'npy', 'rb') as rfb:
                fb = np.load(rfb)
        except:
            fb = np.load(fn[:-3] + 'npz')['features']
    except IOError:
        raise
    return fb

pairs = []
same_spkrs = 0
diff_spkrs = 0
with open(sys.argv[1]) as rf:
    cword = ''
    fs = []
    for line in rf:
        l = line.rstrip('\n')
        if l == '':
            continue
        if "Class" in line:
            cword = l.split()[1]
            fs = []
        else:
            fname, start, end = l.split()
            start = int(float(start) * FBANKS_RATE)
            end = int(float(end) * FBANKS_RATE)
            tmp = do_fbank(fname)[start:end+1]
            for (fname2, tmp2) in fs:
                dtw = DTW(tmp, tmp2, return_alignment=1)
                if dataset == 'english':
                    spkr1 = fname[:3]
                    spkr2 = fname2[:3]
                elif dataset == 'xitsonga':
                    spkr1 = fname[-9:-6]
                    spkr2 = fname2[-9:-6]
                else:
                    raise IOError
                if spkr1 == spkr2:
                    same_spkrs += 1
                else:
                    diff_spkrs += 1
                pairs.append((cword, spkr1, spkr2, tmp, tmp2, dtw[0], dtw[-1][1], dtw[-1][2]))
            fs.append((fname, tmp))
joblib.dump(pairs, sys.argv[2],
            compress=3, cache_size=512)
print "ratio same spkrs / all:", float(same_spkrs) / (same_spkrs + diff_spkrs)
