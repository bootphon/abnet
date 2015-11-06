import sys
from dtw import DTW
import numpy as np
import joblib, glob, os, sys
from spectral import Spectral
from scipy.io import wavfile
import h5features
import operator
from abnet.utils.stack_fbanks import stack_fbanks
# MAX_LENGTH_WORDS = 6     # in phones
# MIN_LENGTH_WORDS = 6     # in phones
# MIN_FRAMES = 5           # in speech frames
# FBANKS_TIME_STEP = 0.01  # in seconds
FBANKS_WINDOW = 0.025    # 25ms
FBANKS_RATE = 100        # 10ms
N_FBANKS = 40  # number of filterbanks to use

#bdir = '/fhgfs/bootphon/scratch/gsynnaeve/zerospeech/english_wavs/'
#bdir = '/fhgfs/bootphon/scratch/roland/abnet/buckeye_mdeltas/'
#bdir = '/fhgfs/bootphon/scratch/edunbar/buckeye_fb/fb_vadnorm/'


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


# @Memoize
def do_fbank(fname):
    fn = fname + '.wav'
    srate, sound = wavfile.read(fn)
    fbanks = Spectral(nfilt=N_FBANKS,    # nb of filters in mel bank
                 alpha=0.97,             # pre-emphasis
                 do_dct=False,           # we do not want MFCCs
                 fs=srate,               # sampling rate
                 frate=FBANKS_RATE,      # frame rate
                 wlen=FBANKS_WINDOW,     # window length
                 nfft=1024,              # length of dft
                 do_deltas=False,        # speed
                 do_deltasdeltas=False   # acceleration
                 )
    fb = np.array(fbanks.transform(sound), dtype='float32')
    return fb


def h5features_fbanks(files, h5f, featfunc=do_fbank, timefunc=None):
    batch_size = 500
    features = []
    times = []
    internal_files = []
    i = 0
    for f in files:
        if i == batch_size:
            h5features.write(h5f, '/features/', internal_files, times,
                             features)
            features = []
            times = []
            internal_files = []
            i = 0
        i = i+1
        data = featfunc(f)
        features.append(data)
        if timefunc == None:
            time = np.arange(data.shape[0], dtype=float) * 0.01 + 0.0025
        else:
            time = timefunc(f)
        times.append(time)
        internal_files.append(os.path.basename(os.path.splitext(f)[0]))
    if features:
        h5features.write(h5f, '/features/',
                         internal_files, times,
                         features)


def h5features_feats2stackedfeats(fb_h5f, stackedfb_h5f):
    index = h5features.read_index(fb_h5f)
    def aux(f):
        return stack_fbanks(h5features.read(fb_h5f, from_internal_file=f,
                                            index=index)[1][f])
    def time_f(f):
        return stack_fbanks(h5features.read(fb_h5f, from_internal_file=f,
                                            index=index)[0][f])
    h5features_fbanks(index['files'], stackedfb_h5f, featfunc=aux,
                      timefunc=time_f)


def run(fin, fout, h5f, verbose=0, from_features=False):
    """main function
    
    fin: same pairs file
        (word_d, wav1, start1, stop1, wav2, start2, stop2, spk1, spk2)
    fout: joblib file
    h5f: h5features file to compute or use
    from_features: use precomputed features file"""
    wavs = set()
    pairs = []
    same_spkrs = 0
    diff_spkrs = 0
    if verbose:
        print('Computing filterbanks')
    if not from_features:
        with open(fin) as rf:
            for line in rf:
                _, wav1, _, _, wav2, _, _, _, _ = line.strip().split()
                wavs.add(wav1)
                wavs.add(wav2)
        h5features_fbanks(wavs, h5f)
    index = h5features.read_index(h5f)

    with open(fin) as rf:
        for line in rf:
            (word, wav1, start1, stop1,
             wav2, start2, stop2, spk1, spk2) = line.strip().split()
            start1, stop1, start2, stop2 = map(float,
                                               (start1, stop1, start2, stop2))
            file1, file2 = map(operator.itemgetter(0),
                               map(os.path.splitext,
                                   map(os.path.basename, (wav1, wav2))))
            feat1 = h5features.read(h5f, from_internal_file=file1,
                                    from_time=start1, to_time=stop1,
                                    index=index)[1][file1]
            feat2 = h5features.read(h5f, from_internal_file=file2,
                                    from_time=start2, to_time=stop2,
                                    index=index)[1][file2]
            dtw = DTW(feat1, feat2, return_alignment=1)
            pairs.append((word, spk1, spk2,
                          feat1,
                          feat2,
                          dtw[0], dtw[-1][1], dtw[-1][2]))
            # pairs.append((None, spk1, spk2,
            #               (wav1, start1, stop1),
            #               (wav2, start2, stop2),
            #               dtw[0], dtw[-1][1], dtw[-1][2]))
            if spk1 == spk2:
                same_spkrs += 1
            else:
                diff_spkrs += 1
    joblib.dump(pairs, fout, compress=3, cache_size=512)
    if verbose:
        print("ratio same spkrs / all:",
              float(same_spkrs) / (same_spkrs + diff_spkrs))


if __name__ == '__main__':
    fin = sys.argv[1]
    fout = sys.argv[2]
    h5f = sys.argv[3]

    run(fin, fout, h5f)

