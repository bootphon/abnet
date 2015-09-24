"""python stack_fbanks.py npz11_train/*.npy
"""

import sys
import numpy as np
import os

NFRAMES = 7
b_a = (NFRAMES - 1) / 2
FRAMES_PER_SEC = 100  # features frames per second
FEATURES_RATE = 1. / FRAMES_PER_SEC

try:
    os.makedirs(sys.argv[1])
except OSError:
    pass

for fname in sys.argv[2:]:
    print fname
    fbanks = np.load(fname)
    fbanks7 = np.zeros((fbanks.shape[0], fbanks.shape[1] * NFRAMES),
            dtype='float32')
    for i in xrange(fbanks.shape[0]):
        fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
                (max(0, (b_a - i) * fbanks.shape[1]),
                    max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
                'constant', constant_values=(0, 0))
#    for i in xrange(b_a + 1):
#        fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
#                (max(0, (b_a - i) * fbanks.shape[1]),
#                    max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
#                'constant', constant_values=(0, 0))
#    for i in xrange(b_a + 1, fbanks.shape[0] - b_a):
#        fbanks7[i] = fbanks[i - b_a:i + b_a + 1].flatten()
#    for i in xrange(fbanks.shape[0] - b_a - 1, fbanks.shape[0]):
#        fbanks7[i] = np.pad(fbanks[max(0, i - b_a):i + b_a + 1].flatten(),
#                (max(0, (b_a - i) * fbanks.shape[1]),
#                    max(0, ((i+b_a+1) - fbanks.shape[0]) * fbanks.shape[1])),
#                'constant', constant_values=(0, 0))
    time_table = np.zeros(fbanks7.shape[0])
    for i in xrange(time_table.shape[0]):
        time_table[i] = float(i) / FRAMES_PER_SEC + FEATURES_RATE / 2
    np.savez(os.path.join(sys.argv[1], os.path.basename(fname).split('.')[0] + '.npz'),
            features=fbanks7,
            time=time_table)

