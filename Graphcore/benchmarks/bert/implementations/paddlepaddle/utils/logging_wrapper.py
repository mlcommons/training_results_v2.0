# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os


class SimpleWrapper:
    def __init__(self, args, mllog):

        # Only log on one instance (if in distributed mode)
        self.should_log = (args.POD < 128 or args.popdist_rank == 0)
        if self.should_log:
            self.mllogger = mllog.get_mllogger()
            filename = f"results/ipu-pod{args.POD}-PaddlePaddle-closed/bert/result_{args.submission_run_index}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if os.path.exists(filename):
                os.remove(filename)
            mllog.config(filename=filename)

    def start(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.start(*args, **kwargs)

    def end(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.end(*args, **kwargs)

    def event(self, *args, **kwargs):
        if self.should_log:
            self.mllogger.event(*args, **kwargs)
