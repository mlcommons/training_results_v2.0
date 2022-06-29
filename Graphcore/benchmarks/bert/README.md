# MLPerf BERT Training on IPUs

This readme describes how to run the benchmarks.
This code runs with Poplar SDK 2.3.
Install the SDK, the requirement.txt and the `mlperf-logging` packaged as described in the `implementations/popart` README..
The pretraining dataset has to be setup as described in the README at `implementations/popart`

To run the benchmark, one has to define the following environment variables like:

```
export HOSTS=<PROVIDED_HOST_IPS>
export VIPU_SERVER_HOST=<PROVIDED_VIPU_SERVER_HOST>
export PARTITION_NAME=<PROVIDED_PARTITION_NAME>
export CLUSTER_NAME=<PROVIDED_CLUSTER_NAME>
```

Then set the `POPLAR_ROOT` and related variables by doing:

```
export POPLAR_ROOT=<LOCATION_OF_POPLAR_LIBRARY>
export OMPI_CPPFLAGS=${POPLAR_ROOT}/include/openmpi/
export OPAL_PREFIX=${POPLAR_ROOT}
```

For running the benchmark for the closed division for the different
accelerator sizes (16, 64, 128),
execute:

```
./submit_closed.sh NUM_ACCELERATORS
```

and for running the benchmarks for the open division, do:

```
./submit_open.sh NUM_ACCELERATORS
```

The result files will be in the respective logging folders of each run.
The parameter configurations can be found in `configs/mk2`.
