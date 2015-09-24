import subprocess
import sys
import numpy as np
import cPickle
import glob
from nnet_archs import ABNeuralNet2Outputs
import tempfile
import os
import random
import string
import h5features
import pandas
import ast
import itertools
# import ABXpy.task as task
# import ABXpy.score as score
# import ABXpy.analyze as analyze
# import ABXpy.distances.distances as distances
# import ABXpy.distances.metrics.dtw as dtw
# import ABXpy.distances.metrics.cosine as cosine
# import distance


NFEATURES = 100

# pickles = map(str, ['best', 10, 20, 40, 80, 200, 500])
# depths = [1, 3, 5]
# layers = ['SigmoidLayer', 'ReLU']
# d_hidden = [500, 1000]
# pickles = ["_".join(['english_mfcc/plp', 'fbank7', 'adadelta', str(archi[0]), archi[1], str(archi[2]), 'emb100']) for archi in itertools.product(depths, layers, d_hidden)]
# pickles = ['tests/plp_fbank7_AB_adadelta_emb_100']
# pickles = ['best']
# pickles = [x + 'plp_fbank7_AB_adadelta_emb_100' for x in ['xitsonga_plp_mdf25/', 'xitsonga_plp_mdf50/']]
# pickles = [x + 'plp_fbank7_AB_adadelta_emb_100' for x in ['english_500_cut_01403/', 'english_500_cut_02306/']]
# pickles = ['mfcc_gmm']
# pickles = ['english_fb/plp_fbank7_AB_adadelta_emb_100_test']
# pickles = ['english_plp_mdf25/plp_fbank7_AB_adadelta_emb_100', 'english_plp_mdf50/plp_fbank7_AB_adadelta_emb_100']
nslots=10
abxpybin = '/home/roland/ABXpy/ABXpy'


# within_task = 'english_within.abx'
# across_task = 'english_across.abx'
# data = 'english_fb/data'
# data = 'english_wrdemb/stacked_data'

# within_task = '/home/roland/english_smaller_within.abx'
# across_task = '/home/roland/english_smaller_across.abx'
# data = 'english_fb/data'
# mean_std_file = 'english_fb/plp_mean_std.npz'

within_task = 'xitsonga_within.abx'
across_task = 'xitsonga_across.abx'
mean_std_file = 'xitsonga_fb/fb_mean_std.npz'
data = 'xitsonga_fb/data'


def default_distance(x, y):
    """ Dynamic time warping cosine distance

    The "feature" dimension is along the columns and the "time" dimension
    along the lines of arrays x and y
    """
    if x.shape[0] > 0 and y.shape[0] > 0:
        # x and y are not empty
        d = dtw.dtw(x, y, cosine.cosine_distance)
    elif x.shape[0] == y.shape[0]:
        # both x and y are empty
        d = 0
    else:
        # x or y is empty
        d = np.inf
    return d


xitsonga_spk_list = {
    "130m", "139f", "132m", "102f",
    "128m", "103f", "146f", "134m",
    "104f", "135m", "141m", "001m",
    "142m", "131f", "126f", "143m",
    "138m", "127f", "144m", "133f",
    "145m", "129f", "140f", "136f",
}


def get_distance(distance_path):
    distancepair = distance_path.split('.')
    distancemodule = distancepair[0]
    distancefunction = distancepair[1]
    path, mod = os.path.split(distancemodule)
    sys.path.insert(0, path)
    distancefun = getattr(__import__(mod), distancefunction)
    return distancefun


def evaluate(abnet_pickle, stackedfbanks, mean_std_file, h5features_file):
    with open(abnet_pickle, 'rb') as f:
        nnet = cPickle.load(f)

    NFRAMES = 7#nnet.layers_ins[0] / NFEATURES
    in_fldr = stackedfbanks
    transform = nnet.transform_x1()
    tmp = np.load(mean_std_file)
    mean = np.concatenate((np.tile(tmp['mean'], NFRAMES), [0]*20))
    std = np.concatenate((np.tile(tmp['std'], NFRAMES), [1]*20))
    # mean = np.concatenate((np.tile(tmp['mean'], NFRAMES), [0]*20))
    # std = np.concatenate((np.tile(tmp['std'], NFRAMES), [1]*20))
    mean = np.tile(tmp['mean'], NFRAMES)
    std = np.tile(tmp['std'], NFRAMES)

    # TODO maybe normalize embedded features ???
    for fname in glob.iglob(os.path.join(in_fldr, "*.npz")):
        if 'talker' in fname or ('nchlt' in fname and not any([x in fname for x in xitsonga_spk_list])):
            continue
        npz = np.load(fname)
        X = np.asarray((npz['features'] - mean) / std, dtype='float32')
        times = npz['time']
        # times = np.arange(0.01, 0.01*npz.shape[0], 0.01)
        emb_wrd, emb_spkr = transform(X)
        h5features.write(h5features_file, '/features/', [os.path.splitext(os.path.basename(fname))[0]], [times], [emb_wrd])
        print("did " + fname)


TEMPLATE_SERIAL = """
#####################################
#$ -S /bin/bash
#$ -cwd
#$ -N {name}
#$ -j y
#$ -o {logfile}
#$ -q {queue}
#$ -pe openmpi_ib {slots}
#$ -V
#####################################
echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
. /etc/profile.d/modules.sh
export PYTHONPATH=/fhgfs/bootphon/scratch/roland/abnet:$PYTHONPATH
{script}
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"
"""


def random_id(length=8):
    return ''.join(random.sample(string.ascii_letters + string.digits, length)) 


def submit_python_code(code, name="job", logfile="output.$JOB_ID", cleanup=True, prefix="", slots=1, queue="all.q", holds=[]):
    base = prefix + "submit_{0}".format(random_id())
    open(base + '.py', 'wb').write(code)
    script = "python " + base + ".py"
    qholds = ''
    if holds:
        qholds = '-hold_jid ' + ','.join(holds) + ' '
    if queue == 'gpu':
        script = 'module load cuda55;\n' + script
    open(base + '.qsub', 'wb').write(TEMPLATE_SERIAL.format(script=script, name=name, logfile=logfile, slots=slots, queue=queue))
    try:
        ret = subprocess.check_output('qsub ' + qholds + base + '.qsub', shell=True)
        print ret
    finally:
        if cleanup:
            os.remove(base + '.qsub') 
    return ret.split()[2]


def avg(filename, task_type):
    df = pandas.read_csv(filename, sep='\t')
    if task_type=='across':
        df['context'] = df['by']
    elif task_type=='within':
        arr = np.array(map(ast.literal_eval, df['by']))
        df['talker']  = [e for e, f in arr]
        df['context'] = [f for e, f in arr]
    else:
        raise ValueError('Unknown task type: {0}'.format(task_type))
    del df['by']
    # aggregate on talkers
    groups = df.groupby(['context', 'phone_1', 'phone_2'], as_index=False)
    df = groups['score'].mean()
    # aggregate on contexts    
    groups = df.groupby(['phone_1', 'phone_2'], as_index=False) 
    df = groups['score'].mean()
    return df.mean()[0]


def run():
    pickles = [0]
    for p in pickles:
        # abnet_file = p + '.pickle'
        # abnet_file = os.path.join('english_mfcc', 'plp_fbank7_AB_adadelta_emb_100_' + p + '.pickle')
        # abnet_file = os.path.join('xitsonga_fb_2', 'plp_fbank7_AB_adadelta_emb_100_' + p + '.pickle')
        abnet_file = os.path.join('xitsonga_fb_2', 'fb_fbank7_adadelta_3_SigmoidLayer_500_emb100.pickle')
        if not os.path.exists(abnet_file):
            continue

        mean_std_file = os.path.join(os.path.dirname(abnet_file), 'fb_mean_std.npz')
        h5file = os.path.splitext(abnet_file)[0] + '.h5f'
        dist_within = os.path.splitext(abnet_file)[0] + '_within.dist'
        dist_across = os.path.splitext(abnet_file)[0] + '_across.dist'
        score_within = os.path.splitext(abnet_file)[0] + '_within.score'
        score_across = os.path.splitext(abnet_file)[0] + '_across.score'
        csv_within = os.path.splitext(abnet_file)[0] + '_within.csv'
        csv_across = os.path.splitext(abnet_file)[0] + '_across.csv'

        w_holds = []
        c_holds = []

        evaluate_code = """
from abnet_abx import evaluate
evaluate('{}', '{}', '{}', '{}')
""".format(abnet_file, data, mean_std_file, h5file)
        holds = submit_python_code(evaluate_code, name='transform', queue='gpu')
        w_holds = [holds]
        c_holds = [holds]

        compute_within_code = """
import compute_distances
compute_distances.run('{}', '{}', '{}', '{}', '{}')
""".format(h5file, within_task, dist_within, 'distance.kl_divergence', nslots)
        w_holds.append(submit_python_code(compute_within_code, name='distances', slots=nslots, holds=w_holds))

        compute_across_code = """
import compute_distances
compute_distances.run('{}', '{}', '{}', '{}', '{}')
""".format(h5file, across_task, dist_across, 'distance.kl_divergence', nslots)
        c_holds.append(submit_python_code(compute_across_code, name='distances', slots=nslots, holds=c_holds))


        score_within_code = """
import ABXpy.score as score
score.score('{}', '{}', '{}')
""".format(within_task, dist_within, score_within)
        w_holds.append(submit_python_code(score_within_code, name='score', slots=1, holds=w_holds))

        score_across_code = """
import ABXpy.score as score
score.score('{}', '{}', '{}')
""".format(across_task, dist_across, score_across)
        c_holds.append(submit_python_code(score_across_code, name='score', slots=1, holds=c_holds))


        analyze_within_code = """
import ABXpy.analyze as analyze
analyze.analyze('{}', '{}', '{}')
""".format(within_task, score_within, csv_within)
        w_holds.append(submit_python_code(analyze_within_code, name='analyze', slots=1, holds=w_holds))

        analyze_across_code = """
import ABXpy.analyze as analyze
analyze.analyze('{}', '{}', '{}')
""".format(across_task, score_across, csv_across)
        c_holds.append(submit_python_code(analyze_across_code, name='analyze', slots=1, holds=c_holds))
            
        # print p
        # try:
        #     print (1 - avg(csv_across, 'across')) * 100
        #     # print (1 - avg(csv_within, 'within')) * 100
        # except:
        #     print 'no results'
        # print ''

if __name__ == '__main__':
    run()
