#!/bin/bash 

set -x 

source config_DSS8440x8xA30.sh

CONT=b89e1443c5cd ./run_with_docker.sh
