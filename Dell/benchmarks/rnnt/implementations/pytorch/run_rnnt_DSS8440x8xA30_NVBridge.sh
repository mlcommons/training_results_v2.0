#!/bin/bash
set -x

source config_DSS8440x8xA30_NVBridge.sh
NEXP=10
CONT=313c98833a30 ./run_with_docker.sh
