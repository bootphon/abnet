import os
import sys
import cPickle
import numpy as np
import h5py
import shutil
import ABXpy.h5tools.h52np as h52np
import ABXpy.h5tools.np2h5 as np2h5


def evaluate(abnet_pickle, stackedfbanks, mean_std_file, h5features_file, nframes=7):
    with open(abnet_pickle, 'rb') as f:
        print 'loading', f
        nnet = cPickle.load(f)

    try:
        #TODO: replace by copy all but features
        shutil.copy(stackedfbanks, h5features_file)
        del h5py.File(h5features_file)['features']['features']
        NFRAMES = nframes
        transform = nnet.transform_x1()
        #transform = nnet.transform_x1_x2()
        tmp = np.load(mean_std_file)
        mean = tmp['mean']#np.tile(tmp['mean'], NFRAMES)
        std = tmp['std']#np.tile(tmp['std'], NFRAMES)
        embedding_size = nnet.layers_outs[-1]

        with h52np.H52NP(stackedfbanks) as f_in, \
             np2h5.NP2H5(h5features_file) as f_out:
            inp = f_in.add_dataset('features', 'features', buf_size=10000)
            out = f_out.add_dataset(
                'features', 'features', buf_size=10000,
                n_rows=inp.n_rows, n_columns=embedding_size,
                item_type=np.float32)
            for X in inp:
                print X.shape
                X = (X.astype(np.float32) - mean) / std
                #emb_wrd, emb_spkr = transform(X)
                emb_wrd = transform(X)
                print emb_wrd.shape
                out.write(emb_wrd)

        # X = h5py.File(stackedfbanks)['features']['features'][...]
        # X = (X - mean) / std
        # emb_wrd, emb_spkr = transform(X)
        # h5py.File(h5features_file)['features']['features'] = emb_wrd
    except:
        os.remove(h5features_file)
        raise

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
