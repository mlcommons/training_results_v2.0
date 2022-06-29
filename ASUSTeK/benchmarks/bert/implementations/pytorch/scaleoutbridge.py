import os
from time import time
from collections import defaultdict
import torch
from mlperf_logger import log_event

class Metricstats(object):
    def __init__(self):
        self.total = 0
        self.count = 0
        self.min = 1000000
        self.max = 0
    def addtag(self, dur):
        self.total += dur
        self.count += 1
        if dur < self.min:
           self.min = dur
        if dur > self.max:
           self.max = dur
    def getstats(self):
        return self.total, self.count, self.min, self.max
    def getcount(self):
        return self.count

class ScaleoutBridge(object):
    FWD_TIME = 'fwd_time'
    BWD_TIME = 'bwd_time'
    FWD_BWD_TIME = 'fwd_bwd_time'
    OPT_TIME = 'opt_time'
    LOAD_TIME = 'load_time'
    EVAL_TIME = 'eval_time'
    ITER_TIME = 'iter_time'
    EPOCH_TIME = 'epoch_time'

    def __init__(self, qmax, time_tags, nvtx_flag, deviceid):
        print("Scaleout performance bridge is running ...")
        self.qmax = qmax
        self.time_tags = time_tags
        self.nvtx_flag = nvtx_flag
        self.deviceid = deviceid
        self.bridgestats = defaultdict(Metricstats)
        self.start_epoch = 0
        self.start_eval = 0

        '''tracking one tag at a time'''
        self.start_time = 0

    def push_nvtx(self, tag):
        torch.cuda.nvtx.range_push(tag)

    def pop_nvtx(self):
        torch.cuda.nvtx.range_pop()

    def print_tag(self, tag, dur):
        log_event(key=tag, value={'r':self.deviceid, 't':dur}, log_all_ranks=True, sync=False)

    def add_tag(self, tag, dur):
        self.bridgestats[tag].addtag(dur)
        if tag == self.OPT_TIME:
            if self.bridgestats[tag].getcount() > self.qmax:
                self.printstats()
                return 0
        return 1

    def start_prof(self, tag):
        if self.time_tags:
            torch.cuda.synchronize()
            self.start_time = time()
        if self.nvtx_flag:
            self.push_nvtx(tag)

    def stop_prof(self, tag):
        if self.time_tags:
            torch.cuda.synchronize()
            if not self.add_tag(tag, time()-self.start_time):
                self.printstats()
                self.time_tags = 0
            self.start_time = 0
        if self.nvtx_flag:
            self.pop_nvtx()
        return self.time_tags

    def stop_start_prof(self, tag1, tag2):
        if self.time_tags:
            torch.cuda.synchronize()
            new_start_time = time()
            if not self.add_tag(tag1, new_start_time-self.start_time):
                self.printstats()
                self.time_tags = 0
            self.start_time = new_start_time
        if self.nvtx_flag:
            self.pop_nvtx()
            self.push_nvtx(tag2)

    def start_epoch_prof(self):
        torch.cuda.synchronize()
        self.start_epoch = time()

    def stop_epoch_prof(self):
        self.printstats()
        torch.cuda.synchronize()
        self.print_tag(self.EPOCH_TIME, time()-self.start_epoch)

    def start_eval_prof(self):
        torch.cuda.synchronize()
        self.start_eval = time()

    def stop_eval_prof(self):
        self.printstats()
        torch.cuda.synchronize()
        self.print_tag(self.EVAL_TIME, time()-self.start_eval)

    def printstats(self):
        if not self.time_tags:
            return
        for tag in self.bridgestats:
            self.printstat(tag)
        self.bridgestats.clear()

    def printstat(self, tag):
        total, count, minimum, maximum = self.bridgestats[tag].getstats()
        log_event(key=tag+"_total", value={'r':self.deviceid, 't':total}, log_all_ranks=True, sync=False)
        log_event(key=tag+"_count", value={'r':self.deviceid, 't':count}, log_all_ranks=True, sync=False)
        log_event(key=tag+"_min", value={'r':self.deviceid, 't':minimum}, log_all_ranks=True, sync=False)
        log_event(key=tag+"_max", value={'r':self.deviceid, 't':maximum}, log_all_ranks=True, sync=False)

class EmptyObject(object):
    def start_prof(self, tag):
        pass
    def stop_prof(self, tag):
        pass
    def stop_start_prof(self, tag1, tag2):
        pass
    def start_epoch_prof(self):
        pass
    def stop_epoch_prof(self):
        pass
    def start_eval_prof(self):
        pass
    def stop_eval_prof(self):
        pass

class ScaleoutBridge_Epoch(object):
    def __init__(self, deviceid):
        print("Scaleout performance bridge-epoch only is running ...")
        self.start_time = 0
        self.deviceid = deviceid
    def start_prof(self, tag):
        pass
    def stop_prof(self, tag):
        pass
    def stop_start_prof(self, tag1, tag2):
        pass
    def start_epoch_prof(self):
        torch.cuda.synchronize()
        self.start_time = time()
    def stop_epoch_prof(self):
        torch.cuda.synchronize()
        log_event(key='epoch_time', value={'r':self.deviceid, 't':time()-self.start_time}, log_all_ranks=True, sync=False)

def init_bridge(deviceid):
    time_tags = int(os.getenv('TIME_TAGS', 0))
    nvtx_flag = int(os.getenv('NVTX_FLAG', 0))
    epoch_only = int(os.getenv('EPOCH_PROF', 0))
    sbridge = EmptyObject()
    if time_tags or nvtx_flag:
        sbridge = ScaleoutBridge(1000, time_tags, nvtx_flag, deviceid)
    elif epoch_only:
        sbridge = ScaleoutBridge_Epoch(deviceid)
    return sbridge
