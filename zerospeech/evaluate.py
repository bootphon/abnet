import os
import sys
from abnet.nnet_archs import ABNeuralNet2Outputs
import cPickle
import glob
import numpy as np
import sys
import h5features


def evaluate(abnet_pickle, stackedfbanks, mean_std_file, h5features_file, output="hf5"):
    if output=="txt":
        try:
            os.makedirs(h5features_file)
        except OSError:
            pass
    with open(abnet_pickle, 'rb') as f:
        nnet = cPickle.load(f)

    in_fldr = stackedfbanks
    NFRAMES = 7
    transform = nnet.transform_x1()
    tmp = np.load(mean_std_file)
    mean = np.tile(tmp['mean'], NFRAMES)
    std = np.tile(tmp['std'], NFRAMES)

    # TODO maybe normalize embedded features ???
    for fname in glob.iglob(os.path.join(in_fldr, "*.npz")):
        if 'talker' in fname:
            continue
        npz = np.load(fname)
        X = np.asarray((npz['features'] - mean) / std, dtype='float32')
        times = npz['time']
        # times = np.arange(0.01, 0.01*npz.shape[0], 0.01)
        emb_wrd, emb_spkr = transform(X)
        if output == "h5f":
            h5features.write(h5features_file, '/features/', [os.path.splitext(os.path.basename(fname))[0]], [times], [emb_wrd])
        else:
            np.savetxt(os.path.join(h5features_file, os.path.splitext(os.path.basename(fname))[0]),
                       np.hstack((times[:, np.newaxis], emb_wrd)), '%.4f')
        print("did " + fname)

if __name__ == '__main__':
    try:
        evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
                 sys.argv[5])
    except:
        evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
