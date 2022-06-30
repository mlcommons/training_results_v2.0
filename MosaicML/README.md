# MLPerf Training v2.0 MosaicML Submission

Repository for [MosaicML](http://www.mosaicml.com)'s submission to the MLperf Training v2.0 Open Division benchmark. This submission has several goals:
* Submit a PyTorch-based ResNet-50 using an easy-to-use trainer, our open-source [Composer](https://github.com/mosaicml/composer) library.
* Highlight the gains from our recipes of methods that speed up the training of deep learning models by changing the training algorithm.

# Submitted Benchmarks

We submitted the [ResNet-50](https://github.com/mlcommons/training/tree/master/image_classification) 
trained on [ImageNet](http://image-net.org/) benchmark using [PyTorch](http://pytorch.org) with two configurations:
* **baseline**: A ResNet-50 baseline with label smoothing and using the channels last optimization. The optimizer and learning rate schedules use SGD and learning rate warmup as described in [Goyal et al, 2017](https://arxiv.org/pdf/1706.02677.pdf). 
* **methods**: Without changing the optimizer and learning rate hyperparameters, we modify the training algorithm with several speed up metohds from our [Composer](https://github.com/mosaicml/composer) library.

Each configuration is trained using our open source library [Composer](https://github.com/mosaicml/composer) on 8x NVIDIA A100-80GB GPUs on a single server. The library includes an `MLPerfCallback` that generates the results logs; this will make it easier for the broader community to submit to the MLPerf Open Division in the future.

All submissions use the excellent [FFCV](https://github.com/libffcv/ffcv) dataloader. 

# Configuration details

Each configuration is defined with a YAML file, linked in the below table.

| Benchmark | Benchmark Config | Description | Speedup Methods |
| --- | --- | --- | --- |
| resnet | [8xA100_80GB-baseline](benchmarks/resnet/implementations/8xA100_80GB-baseline/config.yaml) | Base training recipe | Channels Last, Decoupled Weight Decay, Label Smoothing |
| resnet | [8xA100_80GB-methods](benchmarks/resnet/implementations/8xA100_80GB-methods/config.yaml) | Optimized training recipe | Binary Cross Entropy Loss, Blurpool, Channels Last, Decoupled Weight Decay, Exponential Moving Average, FixRes, Label Smoothing, Progressive Resizing, Scale Schedule Ratio |

For details on our speed up methods, please see our [Methods Overview](https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html) documentation.

# System Overview 

These benchmarks have been tested with the following machine configuration:

* 2x AMD EPYC 7513 32-Core Processor
* 8x NVIDIA A100-SXM4-80GB GPUs

# Reproducing Results

To exactly reproduce our results, following the instructions in the [implementations](benchmarks/resnet/implementations/composer) folder to setup and run the `run_and_time.sh` script. We use YAML along with a config manager [yahp](https://github.com/mosaicml/yahp).

## Using algorithms in your own code

One of our goals is enabling the community to access these optimizations outside of MLPerf benchmarks, and to easily submit your own configs to MLPerf. While this submission code uses YAML and scripting, these speed-up methos can also be applied directly to your own python code. For example, this submission's speed-up recipe can be applied with:

```python
from composer import Trainer, algorithms
from composer.callbacks import MLPerfCallback

trainer = Trainer(
    model=your_classification_model,
    train_dataloader=your_dataloader,
    optimizers=your_optimizer,
    schedulers=your_scheduler,
    
    # speed-up algorithms below
    algorithms=[
        algorithms.BlurPool(),
        algorithms.ChannelsLast(),
        algorithms.LabelSmoothing(
            alpha=0.08,
        ),
        algorithms.ProgressiveResizing(
            size_increment=4,
            delay_fraction=0.4
        ),
        algorithms.EMA(
            update_interval='20ba',
        )
    ],
    
    # optional: MLperf logging
    callbacks=[
        MLPerfCallback('/results/', index=0)
    ],
    ...
)

trainer.fit()

```

For more details, see the Composer [documentation](https://docs.mosaicml.com/en/stable/) and the [MLPerfCallback](https://docs.mosaicml.com/en/stable/api_reference/composer.callbacks.mlperf.html#composer.callbacks.mlperf.MLPerfCallback)

## Software Packages

* [Composer](https://github.com/mosaicml/composer)
* MosaicML's [PyTorch Vision](https://hub.docker.com/r/mosaicml/pytorch_vision/tags) Docker Image
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [FFCV](https://github.com/libffcv/ffcv)

# Repository Contents

The repository is submitted with the following:

* `benchmarks`: Configuration files and Composer entrypoint code for running submitted benchmarks
* `results`: Run results for each benchmark
* `systems`: System configuration for each benchmark
