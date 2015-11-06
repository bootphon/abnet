"""Sampling module to sample pair of same/diff frames
and feed it to the abnet

Samples from a a list of pairs and/or temporal distribution and
the features in h5features format
Generate a file providing batch of [features, features, Y] to the abnet
"""
# import collections
from dtw import DTW
import h5features
import numpy as np
import logging
import h5py
import random
                        

logger = logging.getLogger(__name__)


def sample(features_file, output_file,
           data_points=100000, batch_size=100, nframes=7,
           pair_list=None, objective=True,
           prop_words=1, objective_time=True, prop_same_time=0.2,
           prop_same_word=0.5, objective_word=True,
           objective_same=True, objective_diff=True,
           repeat=5, max_epochs=500,):
    """Main sample function

    Sample "phones" same/diff from time information or list of pairs of same word.
    Organize each sample (1 epoch) in small batches for the network

    The sampling is done on 2 entries: the close/far frames (referred to as "time"
    data points) and the same/diff words (referred to as "word" data points)

    The sampling of pairs same/diff works by uniform sample on the list of pairs
    until the objective function is satisfied.

    Parameters:
    -----------
    data_points: int, number of data points per epoch (number of "phones" same + diff)
    batch_size: int, number of data points per batch (optimize for the gpu)
    nframes: int, number of frames to stack as input
    pair_list: str, path to the file containing the list of same word pairs
    objective: general objective function
    prop_words: proportion of data points issued from same/diff words (the rest are
        time information data points)
    objective_time: callable, objective function on the "time" data points
    prop_same_time: float, proportion of same "time" data points
    objective_word: callable, objective function on the "word" data points
    prop_same_word: float, proportion of same "word" data points
    objective_same: callable, objective function on the same "word" data points
    objective_diff: callable, objective function on the diff "word" data points
    repeat: int, use this sample for X epochs before resampling
    """
    # Initialisation
    logger.setLevel(logging.INFO)
    logger.info('Initialisation')
    featuresAPI = FeaturesAPI(features_file)
    if pair_list:
        logger.info('Extracting list of pairs')
        pairs = extract_list(pair_list)
        logger.info('Computing DTWs')
        # do dtw
        margin = (nframes -1) / 2
        aux = []
        for pair in pairs:
            aux.append(list(pair) +
                       list(featuresAPI.do_dtw_withmargin((pair[1], pair[2], pair[3]),
                                                          (pair[4], pair[5], pair[6]),
                                                          margin)))
    pairs = aux
    output = h5py.File(output_file, 'w')
    
    for epoch in range(max_epochs/repeat):
        print epoch
        data = []
        data_time = []
        if prop_words < 1:
            logger.info('Sampling "time" data points')
            #TODO sample from time info
            data_time = sample_time(objective_time, prop_same_time, featuresAPI,
                                    data_points * (1 - prop_words), nframes)

        if prop_words > 0:
            logger.info('Sampling "word" data points')
            n_word_points = data_points * prop_words
            #TODO sample from word info
            if prop_same_word > 0:
                n_same_word_points = n_word_points * prop_same_word
                data_same_word = sample_same_word(objective_same, n_same_word_points,
                                                  featuresAPI, pairs, nframes)
                n_diff_word_points = n_word_points * (1 - prop_same_word)
                data_diff_word = sample_same_word(objective_same, n_diff_word_points,
                                                  featuresAPI, pairs, nframes)

        logger.info('Saving the batches')
        data = [np.concatenate(tup, axis=0)
                for tup in zip(*[l for l in
                                 (data_time, data_same_word, data_diff_word)
                                 if l])]
        output_group = output.create_group(str(epoch))
        output_group.create_dataset('x1', data=data[0])            
        output_group.create_dataset('x2', data=data[1])            
        output_group.create_dataset('y', data=data[2])            


def sample_time(objective, prop_same, featuresAPI, n_data_points, nstacks):
    if prop_same != 0.2:
        logging.error('Only default prop_same_time argument implemented yet')
        raise NotImplementedError
    # Doing it Gabriel's way for now
    SPACES = [20, 30, 40, 50]
    
    files = featuresAPI.index['files']
    x1_tmp, x2_same, x2_diff = [], [], []
    for fname in files:
        features = featuresAPI.get_features_from_file(fname)
        # stacking the features
        stacked = stack_feats(features, nstacks)

        x1_tmp.append(stacked[:-1])
        x2_same.append(stacked[1:])
        for sp in SPACES:
            x2_diff.append(stacked[sp:stacked.shape[0]/2+sp])  # first half shifted by sp
            x2_diff.append(stacked[stacked.shape[0]/2-sp:-sp-1])  # second half shifted by -sp

    # constructing x1, x2 and y
    x1_tmp = np.concatenate(x1_tmp, axis=0)
    nby = (len(SPACES)+1)
    x1 = np.repeat(x1_tmp, nby, axis=0).astype(np.float32)
    x2_same = np.concatenate(x2_same, axis=0)[:, np.newaxis, :]
    x2_diff = np.concatenate(x2_diff, axis=0)
    x2_diff = x2_diff.reshape((-1, len(SPACES), x2_diff.shape[1]))
    x2 = np.asarray(np.concatenate([x2_same, x2_diff], axis=1), dtype='float32')
    x2 = x2.reshape((nby*x2_same.shape[0], -1))  # 'flatten' along the first axis
    y = np.zeros(x1.shape[0], dtype='int32')
    y[::nby] = 1  # 1 for same, 0 for different

    # sanity checks:
    assert all(x1.shape == x2.shape)
    assert x1.shape[0] == y.shape

    # sampling...
    if n_data_points < len(y):
        logging.error('not enough data points found, repeating not implemented yet')
        raise NotImplementedError
    choice = np.random.choice(len(y), n_data_points)
    return x1[choice], x2[choice], y[choice]


def sample_same_word(objective, n_data_points, featuresAPI, pairs, nframes):
    margin = (nframes - 1) / 2
    x1 = []
    x2 = []
    total_len = 0
    random.shuffle(pairs)
    for pair in pairs:
        # segment1 = (pair[1], pair[2], pair[3])
        # segment2 = (pair[4], pair[5], pair[6])
        # dtw_score, dtw_path1, dtw_path2, feat1, feat2 = featuresAPI.do_dtw_withmargin(segment1, segment2, margin)
        dtw_path1, dtw_path2 = pair[10], pair[11]
        feat1, feat2 = pair[12][dtw_path1], pair[13][dtw_path2]
        stacked1 = stack_feats(feat1, nframes)
        stacked2 = stack_feats(feat2, nframes)
        x1.append(stacked1)
        x2.append(stacked2)
        #TODO: review pad, do not redo dtw
        total_len += len(dtw_path1) - 2 * margin
        if total_len > n_data_points:
            break
    x1 = np.concatenate(x1, axis=0)
    x2 = np.concatenate(x2, axis=0)
    assert x1.shape == x2.shape
    y = np.ones((x1.shape[0],))

    # sampling
    if n_data_points < len(y):
        logging.error('not enough data points found, repeating not implemented yet')
        raise NotImplementedError
    choice = np.random.choice(len(y), n_data_points)
    return x1[choice], x2[choice], y[choice]
    
def sample_diff_word(objective, n_data_points, featuresAPI, pairs, nframes):
    margin = (nframes - 1) / 2
    x1 = []
    x2 = []
    total_len = 0
    random.shuffle(pairs)
    for pair1 in pairs:
        # segment1 = (pair1[1], pair1[2], pair1[3])

        # find index so that pairs[index] is a new word
        random_index = np.random.randint(len(pairs))
        while pairs[random_index] == pair1[0]:
            random_index = np.random.randint(len(pairs))            
        pair2 = pairs[random_index]
        # segment2 = (pair2[1], pair2[2], pair2[3])

        feat1, feat2 = pair1[12], pair2[12]
        min_len = min(feat1.shape[0], feat2.shape[0])
        feat1, feat2 = feat1[:min_len], feat2[:min_len]
        
        stacked1 = stack_feats(feat1, nframes)
        stacked2 = stack_feats(feat2, nframes)
        x1.append(stacked1)
        x2.append(stacked2)
        total_len += min_len - 2 * margin
        if total_len > n_data_points:
            break
    x1 = np.concatenate(x1, axis=0)
    x2 = np.concatenate(x2, axis=0)
    assert all(x1.shape == x2.shape)
    y = np.ones((x1.shape[0],))

    # sampling
    if n_data_points < len(y):
        logging.error('not enough data points found, repeating not implemented yet')
        raise NotImplementedError
    choice = np.random.choice(len(y), n_data_points)
    return x1[choice], x2[choice], y[choice]


def stack_feats(features, nframes):
    return np.hstack([features[i:i-nframes]
                      for i in xrange(nframes-1)] + [features[nframes:]])


def extract_list(file_list):
    # Pair = namedtuple('Pair', ['word_id', 'wav1', 'start1', 'end1',
    #                            'wav2', 'start2', 'end2', 'skp1', 'spk2'])
    # class Word_ids:
    #     def __init__(self):
    #         self.data = {}
    #         self.count = 0
            
    #     def index(self, word_id):
    #         if word_id not in self.data:
    #             self.data[word_id] = self.count
    #             self.count += 1
    #         return self.data[word_id]
    # word_ids = Word_ids()
    res = []
    with open(file_list) as fin:
        for line in fin:
            (word_id, wav1, start1, end1,
             wav2, start2, end2, spk1, spk2) = line.strip().split()
            start1, end1, start2, end2 = map(float, (start1, end1, start2, end2))
            res.append([word_id, wav1, start1, end1, wav2, start2, end2, spk1, spk2])
    return res


class FeaturesAPI:
    """wrapper for h5features manipulation
    """
    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.index = h5features.read_index(feature_file)


    def get_features(self, segment):
        """return the features associated to a segment = (file, start, end)"""
        fileid, start, end = segment
        return h5features.read(self.feature_file, from_internal_file=fileid,
                                   from_time=start, to_time=end,
                                   index=self.index)[1][fileid]

    def get_features_from_file(self, fileid):
        """return the features accosiated to a file"""
        return h5features.read(self.feature_file, from_internal_file=fileid,
                               index=self.index)[1][fileid]
    
    def do_dtw(self, segment1, segment2):
        dtw = DTW(self.get_features(segment1), self.get_features(segment2), return_alignment=1)
        return dtw[0], dtw[-1][1], dtw[-1][2]

    def get_features_plusmargin(self, segment, margin):
        """return the features associated to a segment = (file, start, end)
        plus a margin on both side"""
        f1 = self.index['files'].index(segment[0])
        from_time = segment[1]
        to_time = segment[2]
        f_start = self.index['file_index'][f1-1] + 1
        if f1 == 0:
            f_start = 0
        f_end = self.index['file_index'][f1]    
        times = self.index['times'][f_start:f_end+1]

        i1 = f_start + np.where(times>=from_time)[0][0]
        n_zeros_left = max(0, f_start - i1 + margin)
        i1 = max(i1 - margin, f_start)
        times = self.index['times'][f_start:f_end+1] # the end is included...
        i2 = f_start + np.where(times<=to_time)[0][-1]
        n_zeros_right = max(0, i2 - margin - f_end)
        i2 = min(i2 + margin, f_end)
        # Step 2: access actual data
        with h5py.File(self.feature_file, 'r') as f:
            n_dim = f['features']['features'].shape[1]
            pad_left = np.zeros((n_zeros_left, n_dim))
            pad_right = np.zeros((n_zeros_right, n_dim))
            features = np.concatenate((
                pad_left,
                f['features']['features'][i1:i2+1,:],
                pad_right), axis=0)
            times = f['features']['times'][i1:i2+1]
        return features

    def do_dtw_withmargin(self, segment1, segment2, margin):
        feat1 = self.get_features_plusmargin(segment1, margin)
        feat2 = self.get_features_plusmargin(segment2, margin)
        dtw = DTW(feat1[margin:-margin], feat2[margin:-margin], return_alignment=1, python_dist_function=cosine_distance)
        path1_list = dtw[-1][1]
        path1 = np.concatenate((np.arange(margin), np.array(path1_list) + margin,
                                np.arange(path1_list[-1]+1, path1_list[-1]+1 + margin)))
        path2_list = dtw[-1][2]
        path2 = np.concatenate((np.arange(margin), np.array(path2_list) + margin,
                                np.arange(path2_list[-1]+1, path2_list[-1]+1 + margin)))
        assert len(path1) == len(path2), 'path1: {}, {}\npath2:{}, {}'.format(path1, len(path1), path2, len(path2))
        return dtw[0], path1, path2, feat1, feat2


def cosine_distance(A, B):
    return A.T * B / (np.lialg.norm(A) * np.linalg.norm(B))

    
if __name__ == '__main__':
    sample('../../mydev/abx/fb.h5f', 'testf', pair_list='../../mydev/std/pair_list_1000.txt')
