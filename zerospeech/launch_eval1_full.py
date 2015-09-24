import sys
import shutils
import run_exp_STD
from layers import Linear, ReLU, SigmoidLayer, SoftPlus


eval1bin = {
    'english': '/fhgfs/bootphon/scratch/roland/zerospeech2015/english_eval1/eval1',
    'xitsonga': '/fhgfs/bootphon/scratch/roland/zerospeech2015/xitsonga_eval1/eval1',
}

architectures_file = '/fhgfs/bootphon/scratch/roland/abnet/architectures/csv'


def run():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full ABX discrimination task')

    # parser.add_argument(
    #     '-c', '--config', default=os.path.join(curdir, 'resources/sample_eval.cfg'),
    #     help='config file, default to sample_eval.cfg in resources')
    parser.add_argument('classes',
                        help='classes file (input for eval2)')
    parser.add_argument('language',
                        choices=['english', 'xitsonga'],
                        help='language of the dataset')

    args = parser.parse_args()
    run()
