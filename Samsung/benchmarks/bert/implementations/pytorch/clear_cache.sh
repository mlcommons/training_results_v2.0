#!/bin/bash

/home/sr6/zetta/pyenv/versions/CUDA11.4_NGC_PYTORCH/bin/python -u -c "from mlperf_logging.mllog import constants; from mlperf_logger import log_event; log_event(key=constants.CACHE_CLEAR, value=True)"

