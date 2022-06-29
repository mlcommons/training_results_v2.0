from queue import SimpleQueue
from mxnet import cuda_utils as cu
from mlperf_log_utils import mx_resnet_print_event

class ScaleoutBridge(object):
    FWD_TIME = 'fwd_time'
    BWD_TIME = 'bwd_time'
    OPT_TIME = 'opt_time'
    LOAD_TIME = 'load_time'
    EVAL_TIME = 'eval_time'
    EPOCH_TIME = 'epoch_time'
    def __init__(self, qmax):
        print("Scaleout performance bridge is running ...")
        self.qmax = qmax
        self.fwdq = SimpleQueue()
        self.bwdq = SimpleQueue()
        self.optq = SimpleQueue()
        self.loadq = SimpleQueue()
        self.evalq = SimpleQueue()

    def push_nvtx(self, tag):
        cu.nvtx_range_push(tag)

    def pop_nvtx(self):
        cu.nvtx_range_pop()

    def add_tag(self, tag, dur, deviceid):
        if self.fwdq.qsize() > self.qmax:
            self.empty_qs()
            return 0
        if tag == self.FWD_TIME:
            self.fwdq.put((dur, deviceid))
        elif tag == self.BWD_TIME:
            self.bwdq.put((dur, deviceid))
        elif tag == self.OPT_TIME:
            self.optq.put((dur, deviceid))
        elif tag == self.LOAD_TIME:
            self.loadq.put((dur, deviceid))
        elif tag == self.EVAL_TIME:
            self.evalq.put((dur, deviceid))
        else:
            assert("Tag not supported" and 0)
        return 1

    def empty_qs(self):
        self.empty_q(self.fwdq, self.FWD_TIME)
        self.empty_q(self.bwdq, self.BWD_TIME)
        self.empty_q(self.optq, self.OPT_TIME)
        self.empty_q(self.loadq, self.LOAD_TIME)
        self.empty_q(self.evalq, self.EVAL_TIME)

    def empty_q(self, q, tag):
        if q.qsize() >= self.qmax:
            while not q.empty():
                atuple = q.get()
                mx_resnet_print_event(key=tag, val={'r':atuple[1], 't':atuple[0]}, uniq=False)

