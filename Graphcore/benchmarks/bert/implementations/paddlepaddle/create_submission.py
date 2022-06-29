# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import subprocess
import os
import sys
import copy

# Specify which pod to run
if len(sys.argv) != 2:
    raise RuntimeError(
        "Only support 2 args. The second arg should be the number of IPUs.")
pod = sys.argv[1]

current_env = copy.copy(os.environ.copy())

# Each submission consist of 10 runs
print("Launch Bert-L 10-runs with POD{}...".format(pod))
for index in range(0, 10):
    command = f"bash run_mlperf_POD{pod}.sh {index + 43} {index}"

    # Launch the run
    with open(f"internal_log_pod{pod}_{index}", "w+") as f:
        # Run training
        # current_env["POPART_LOG_LEVEL"] = "INFO"
        subprocess.call(
            [command], stdout=f, stderr=f, shell=True, env=current_env)
print("Launch Bert-L 10-runs with POD{}...Done".format(pod))
