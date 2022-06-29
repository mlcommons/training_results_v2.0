import mxnet as mx
import mxnet.cuda_utils as cu
import mlperf_lars

weights = mx.nd.zeros((10,10), ctx=mx.gpu())
mlperf_lars.mlperf_lars_multi_mp_sgd_mom_update(weights, num_weights=1)
