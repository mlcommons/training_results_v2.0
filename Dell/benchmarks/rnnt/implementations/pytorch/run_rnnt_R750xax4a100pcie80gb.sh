#!/bin/bash
set -x 
source config_R750xax4a100pcie80gb.sh
CONT=313c98833a30 ./run_with_docker_node043.sh
