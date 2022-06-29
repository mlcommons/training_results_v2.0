#!/bin/bash

cd ../tensorflow
docker build --pull -t mlperf-inspur:minigo .
