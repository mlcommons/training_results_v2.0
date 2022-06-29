# MLPerf RN50 CNN Training on IPUs

This readme describes how to run the benchmarks. This code runs with Poplar SDK 2.5.
Install the SDK and the `requirement.txt` in the implementations folder.
The TF records have to be located at `/localdata/datasets/imagenet-data`.

For running the benchmarks for the different POD sizes (16, 64, 128, and 256) execute:

```
for ((n=1; n<5; n++)); do
	SEED=$((n << 32));
	run_and_time.sh POD_SIZE $SEED HOSTS VIPU_PARTITION VIPU_SERVER_HOST NETMASK "ON"
done
```

The result files will be in the respective logging folders of each run.
The parameter configurations can be found in `configs.yml` and `run_and_time.sh`.
