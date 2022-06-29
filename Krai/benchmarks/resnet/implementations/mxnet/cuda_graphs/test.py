import mxnet as mx
import mxnet.cuda_utils as cu
import graph_wrapper

a = mx.nd.zeros((10,10), ctx=mx.gpu())
mx.nd.waitall()
mx.nd._internal._plus_scalar(a, out=a, scalar=2)
mx.nd.waitall()
graph_wrapper.start_capture(0, 0, [a])
mx.nd._internal._plus_scalar(a, out=a, scalar=5)
graph_wrapper.end_capture(0, 0, [a])
mx.nd.waitall()
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
graph_wrapper.graph_replay(0, 0, [], [a])
mx.nd.waitall()
print(a)

graph_wrapper.finalize()
