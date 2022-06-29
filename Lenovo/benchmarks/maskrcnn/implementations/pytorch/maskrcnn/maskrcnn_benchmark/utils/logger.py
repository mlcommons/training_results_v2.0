# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
from .comm import is_main_process, is_main_evaluation_process

def setup_logger(name, save_dir, distributed_rank, filename="log.txt", dedicated_evaluation_ranks=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # only master train and master eval ranks log results
    if is_main_process() or is_main_evaluation_process(dedicated_evaluation_ranks):
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
