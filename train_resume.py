import sys
caffe_root = '/home/mcl/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

from pylab import *
import numpy as np


solver_path = 'a14_solver.prototxt'
weights_path = './snapshots/a14_iter_50000.caffemodel'
#---------------------------------------------------------------------------------
# start training

caffe.set_device(0)
caffe.set_mode_gpu()

niter = 10000  # number of iterations to train

solver = caffe.get_solver(solver_path)
solver.net.copy_from(weights_path)

#solver.net.copy_from(weights_path)

for i in range(niter):
    solver.step(1)
