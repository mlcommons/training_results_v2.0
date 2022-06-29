# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import time

import numpy as np
from mpi4py import MPI


def assign_mpiranks(local_rank, size, nodes_for_eval, gpu_per_node):
    # assign top "nodes_for_eval" nodes for evaluation. Rest of the nodes go to training
    local_size = gpu_per_node
    total_ranks = list(range(size))
    train_ranks = total_ranks[:size - nodes_for_eval * gpu_per_node]
    eval_ranks = train_ranks
    transfer_ranks = []
    if nodes_for_eval:
        eval_ranks = total_ranks[size - nodes_for_eval * gpu_per_node:]
        transfer_ranks = [train_ranks[local_rank], *[x for x in eval_ranks if x % local_size == local_rank]]
    assert train_ranks, "Training ranks list is empty"
    assert eval_ranks, "Evaluation ranks list is empty"
    return train_ranks, eval_ranks, transfer_ranks


def get_group_comm(comm, ranks):
    # Create a grouped mpi communicator with the ranks
    # assert len(ranks) > 0, "cannot create group as ranks is empty"
    xcomm = None
    if ranks:
        xgroup = comm.group.Incl(ranks)
        xcomm = comm.Create_group(xgroup)

    return xcomm


def sync_training_and_evaluation(global_comm, transfer_comm, rank, model, eval_ranks, transfer_ranks, stop_training):
    # t0 = time()
    # Let training threads know if evaluation has reached target
    # All reduce also acts as barrier to make sure parameter save is done
    local_stop_training = np.array([stop_training], dtype=np.int32)
    global_stop_training = np.zeros(1, dtype=np.int32)
    global_comm.Allreduce(local_stop_training, global_stop_training, MPI.SUM)
    # t1 = time()

    if rank in transfer_ranks:
        broadcast_model(model, transfer_comm, rank, eval_ranks)

    # Evaluation found end of training
    if global_stop_training != 0:
        stop_training = True

    # if rank == eval_ranks[0] and rank in eval_ranks:
    #     print(f"MODEL TRANSFER TIME t0: {time()-t0}, t1: {time()-t1}")

    return stop_training, model


def broadcast_model(model, comm, rank, eval_ranks):
    params = model._collect_params_with_prefix()

    irequests = []
    result = {}
    for name, p in sorted(params.items()):
        if "dummy" in name:
            continue
        result[name] = p.data().asnumpy()
        irequests.append(comm.Ibcast([result[name], result[name].size * result[name].itemsize, MPI.CHAR], root=0))

    MPI.Request.waitall(irequests)

    if rank in eval_ranks:
        for name, p in sorted(params.items()):
            if "dummy" in name:
                continue
            params[name].set_data(result[name])


def transfer_model(model, global_comm, eval_comm, rank, source_rank, target_rank, eval_ranks):
    params = model._collect_params_with_prefix()

    irequests = []
    result = {}
    for idx, (name, p) in enumerate(sorted(params.items())):
        if "dummy" in name:
            continue
        data = p.data().asnumpy()
        if rank == source_rank:
            irequests.append(global_comm.Isend(data, dest=target_rank, tag=idx))
        elif rank == target_rank:
            result[name] = data
            irequests.append(global_comm.Irecv(result[name], source=source_rank, tag=idx))
        else:
            result[name] = data

    if rank == source_rank:
        MPI.Request.waitall(irequests)

    elif rank in eval_ranks:
        if rank == target_rank:
            MPI.Request.waitall(irequests)
        eval_comm.Barrier()
        for idx, (name, p) in enumerate(sorted(params.items())):
            if "dummy" in name or name not in result.keys():
                continue
            eval_comm.Bcast(result[name], root=0)
            params[name].set_data(result[name])
