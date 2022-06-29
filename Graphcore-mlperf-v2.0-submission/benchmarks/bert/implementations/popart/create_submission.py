# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import argparse
import numpy as np
import subprocess
import os
import re

# Specify which pod to run
parser = argparse.ArgumentParser("Config Parser", add_help=False)
parser.add_argument("--pod", type=int, choices=[16, 64, 128, 256], default=16)
parser.add_argument("--submission-division", type=str, choices=["open", "closed"], default="closed")
parser.add_argument("--start-index", type=int, default=0)
parser.add_argument("--end-index", type=int, default=10)
args = parser.parse_args()

HOSTS = os.getenv('HOSTS')
VIPU_SERVER_HOST = os.getenv('VIPU_SERVER_HOST')
PARTITION_NAME = os.getenv('PARTITION_NAME')
CLUSTER_NAME = os.getenv('CLUSTER_NAME')
TCP_IF_INCLUDE = os.getenv('TCP_IF_INCLUDE')

pod128Command = f"poprun -vv  --remove-partition=0 --num-instances 2 --num-replicas 16  --num-ilds=2 --ipus-per-replica 8 --numa-aware=yes --host {HOSTS} --vipu-server-host={VIPU_SERVER_HOST} --vipu-server-timeout=600 --vipu-partition={PARTITION_NAME} --vipu-cluster={CLUSTER_NAME} --reset-partition=no --update-partition=no --mpi-global-args='--tag-output  --allow-run-as-root  --mca oob_tcp_if_include {TCP_IF_INCLUDE} --mca btl_tcp_if_include {TCP_IF_INCLUDE}' --mpi-local-args='-x SHARED_EXECUTABLE_CACHE -x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=OFF -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=OFF -x POPART_LOG_LEVEL=OFF -x GCL_LOG_LEVEL=OFF' "

pod256Command = f"poprun -vv  --remove-partition=0 --num-instances 4 --num-replicas 32  --num-ilds=4 --ipus-per-replica 8 --numa-aware=yes --host {HOSTS} --vipu-server-host={VIPU_SERVER_HOST} --vipu-server-timeout=600 --vipu-partition={PARTITION_NAME} --vipu-cluster={CLUSTER_NAME} --reset-partition=no --update-partition=no --mpi-global-args='--tag-output  --allow-run-as-root  --mca oob_tcp_if_include {TCP_IF_INCLUDE} --mca btl_tcp_if_include {TCP_IF_INCLUDE}' --mpi-local-args='-x SHARED_EXECUTABLE_CACHE -x LD_LIBRARY_PATH -x OPAL_PREFIX -x PATH -x HOROVOD_STALL_CHECK_TIME_SECONDS=600 -x HOROVOD_LOG_LEVEL=OFF -x HOROVOD_POPART_BROADCAST_TIMEOUT=600 -x CPATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS  -x POPLAR_LOG_LEVEL=OFF -x POPART_LOG_LEVEL=OFF -x GCL_LOG_LEVEL=OFF' "

# Each submission consist of 10 runs
for result_index in range(args.start_index, args.end_index):
    command = f"python bert.py --config=configs/pod{args.pod}-{args.submission_division}.json --seed {result_index + 43}"

    if args.pod == 128:
        command = pod128Command + command
    elif args.pod == 256:
        command = pod256Command + command

    options = f"--submission-run-index={result_index}"

    # Launch the run
    with open(f"pod{args.pod}_internal_log_{result_index}", "w+") as f:
        # Clear the cache
        # subprocess.call(['sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'], stdout=f, stderr=f, shell=True)

        # Run training
        subprocess.call([command + " " + options], stdout=f, stderr=f, shell=True)
