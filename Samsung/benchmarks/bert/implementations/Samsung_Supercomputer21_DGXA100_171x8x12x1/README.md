## Steps to launch training on 171 nodes (totally 1368 GPUs)

### 171 nodes in Samsung Supercomputer21

* 171 nodes configuration and hyperparameters: config_Samsung_Supercomputer21_DGXA100_171x8x12x1.sh
* Launch training of bert for NVIDIA DGX A100 1368 GPUs by the following steps:

  1). Prepare SW requirements listed in [README](../pytorch/README.md) 
  
  2). Set hosts IP addresses in run_multinode.sh
   
  3). Launch the training with 171 nodes
```shell
PYTHON=<path/to/python> DGXSYSTEM=Samsung_Supercomputer21_DGXA100_171x8x12x1 INPUT_DIR=<path/to/4bins_training_datadir> EVAL_DIR=<path/to/eval_datadir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1> NEXP=10 ./run_multinode.sh
```
