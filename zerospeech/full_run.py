import os
import subprocess
import itertools
import random
import string


architectures = []
features = []
classes = []
eval1bin = {
    'english': '/fhgfs/bootphon/scratch/roland/zerospeech2015/english_eval1/eval1',
    'xitsonga': '/fhgfs/bootphon/scratch/roland/zerospeech2015/xitsonga_eval1/eval1',
}


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


def submit_python_code(code, name="job", logfile="output.$JOB_ID", cleanup=True, prefix="", slots=1, queue="all.q"):
    base = prefix + "submit_{0}".format(random_id())
    print code
    open(base + '.py', 'wb').write(code)
    script = "python " + base + ".py"
    if queue == 'gpu':
        script = 'module load cuda55;\n' + script
    open(base + '.qsub', 'wb').write(TEMPLATE_SERIAL.format(script=script, name=name, logfile=logfile, slots=slots, queue=queue))
    try:
        subprocess.call('qsub ' + base + '.qsub', shell=True)
    finally:
        if cleanup:
            os.remove(base + '.qsub') 


def submit_script(script, name="job", logfile="output.$JOB_ID", cleanup=True, prefix="", slots=1, queue="all.q"):
    base = prefix + "submit_{0}".format(random_id())
    if queue == 'gpu':
        script = 'module load cuda55;\n' + script
    open(base + '.qsub', 'wb').write(TEMPLATE_SERIAL.format(script=script, name=name, logfile=logfile, slots=slots, queue=queue))
    try:
        subprocess.call('qsub ' + base + '.qsub', shell=True)
    finally:
        if cleanup:
            os.remove(base + '.qsub')


# class Memoize:
#     """Memoize(fn) 
#     Will only work on functions with non-mutable arguments
#     """
#     def __init__(self, fn):
#         self.fn = fn
#         self.memo = {}
#     def __call__(self, *args):
#         if not self.memo.has_key(args):
#             self.memo[args] = self.fn(*args)
#         return self.memo[args]


# @Memoize
# def do_fbank(fname):
#     with open(os.path.join(bdir, fname + '.npy'), 'rb') as rfb:
#         fb = np.load(rfb)
#     print "did:", fn
#     return fb


# def prepare_joblib(classes_file, joblib_file):
#     pairs = []
#     same_spkrs = 0
#     diff_spkrs = 0
#     with open(classes_file) as rf:
#         cword = ''
#         fs = []
#         for line in rf:
#             l = line.rstrip('\n')
#             if l == '':
#                 continue
#             if "Class" in line:
#                 cword = l.split()[1]
#                 fs = []
#             else:
#                 fname, start, end = l.split()
#                 start = int(float(start) * FBANKS_RATE)
#                 end = int(float(end) * FBANKS_RATE)
#                 tmp = do_fbank(fname)[start:end+1]
#                 for (fname2, tmp2) in fs:
#                     dtw = DTW(tmp, tmp2, return_alignment=1)
#                     spkr1 = fname[:3]
#                     spkr2 = fname2[:3]
#                     if spkr1 == spkr2:
#                         same_spkrs += 1
#                     else:
#                         diff_spkrs += 1
#                     pairs.append((cword, spkr1, spkr2, tmp, tmp2, dtw[0], dtw[-1][1], dtw[-1][2]))
#                 fs.append((fname, tmp))
#     joblib.dump(pairs, joblib_file,
#                 compress=3, cache_size=512)
#     print "ratio same spkrs / all:", float(same_spkrs) / (same_spkrs + diff_spkrs)


xitsonga_spk_list = {
    "130m", "139f", "132m", "102f",
    "128m", "103f", "146f", "134m",
    "104f", "135m", "141m", "001m",
    "142m", "131f", "126f", "143m",
    "138m", "127f", "144m", "133f",
    "145m", "129f", "140f", "136f",
}


def run():

    # dataset_path = 'gold_fb/gold.joblib'
    # dataset_name = 'gold_fb/gold'
    dataset_name = 'xitsonga_fb_2/fb'
    dataset_path = 'xitsonga_fb_2/fb.joblib'
    depths = [3]
    layers = ['SigmoidLayer']#, 'ReLU']
    d_hidden = [500]

    for archi in itertools.product(depths, layers, d_hidden):
        print archi
        l = [archi[1]] * archi[0]
        l = str(l).replace("'", "")
        d = [archi[2]] * (archi[0] - 1)
        output_fname = "_".join([dataset_name, 'fbank7', 'adadelta', str(archi[0]), archi[1], str(archi[2]), 'emb100'])

        run_exp_STD_code = """
import run_exp_STD
from layers import Linear, ReLU, SigmoidLayer, SoftPlus


run_exp_STD.run(dataset_path='{0}', dataset_name='{1}',
    batch_size=100, nframes=7, features='fbank',
    init_lr=0.01, max_epochs=500, 
    network_type='AB', trainer_type='adadelta',
    layers_types={2},
    layers_sizes={3},
    loss='cos_cos2',
    prefix_fname='',
    debug_print=1,
    debug_time=True,
    debug_plot=0,
    mv_file='{1}' + "_mean_std.npz",
    mm_file='{1}' + "_min_max.npz",
    output_file_name='{4}')
""".format(dataset_path, dataset_name, l, d, output_fname)
        submit_python_code(run_exp_STD_code, name='abnet', slots=1, queue='gpu')


if __name__ == '__main__':
    run()
