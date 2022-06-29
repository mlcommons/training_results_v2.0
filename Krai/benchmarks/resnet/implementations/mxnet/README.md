# 1. Problem

The problem scope is image classification. ResNet-50 CNN was used. 

# 2. Running the Benchmark

## Download the dataset

1. Define the directories
```
export DATA_DIR="/data"
export TAR_DIR="${DATA_DIR}/tar"
export TRAIN_DIR="${DATA_DIR}/train"
export TRAIN_TEMP_DIR="${DATA_DIR}/ILSVRC2012_img_train"
export VAL_DIR="${DATA_DIR}/ILSVRC2012_img_val"
export POSTPROCESS_DIR="/data/postprocess"
export DEEP_LEARNING_EXAMPLE_DIR=$HOME
export TRAINING_EXAMPLE_DIR=$(pwd)
export LOG_DIR=$(pwd)/../../../../results
```

2. Download the dataset

* Download **Training images (Task 1 &amp; 2)** and **Validation images (all tasks)** at http://image-net.org/challenges/LSVRC/2012/2012-downloads (require an account), you should get 2 `.tar` files. Store them at `TAR_DIR`
```
mv ILSVRC2012_img_train.tar $TAR_DIR && mv ILSVRC2012_img_val.tar $TAR_DIR
```

3. Extract the data:
* Training Data
```
cd $DATA_DIR && \
tar -xvf $TAR_DIR/ILSVRC2012_img_train.tar && \
find . -name "*.tar" | while read NAME ; do sudo mkdir -p "${TRAIN_DIR}/${NAME%.tar}"; sudo tar -xvf "${NAME}" -C "${TRAIN_DIR}/${NAME%.tar}"; done
```

* Validation Data
```
cd $DATA_DIR && \
tar -xvf $TAR_DIR/ILSVRC2012_img_val.tar && \
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | sudo bash
```

4. You should expect to see:
```
./$DATADIR
│   ├── tar
|   |       ├── ILSVRC2012_img_train.tar
|   |       └── ILSVRC2012_img_val.tar
|   |
│   ├── ILSVRC2012_img_train
|   |       ├── nxxxxx.tar
|   |       ├── nxxxxx.tar
|   |       ...
|   |
│   ├── train
|   |       ├── nxxxxx
│   │       │    ├── xxxxx.JPEG
│   │       │    ....
|   |       ├── nxxxxx
│   │       │    ├── xxxxx.JPEG
│   │       │    ....
|   |       ....
|   |
│   ├── ILSVRC2012_img_val
|   |       ├── nxxxxx
│   │       │    ├── xxxxx.JPEG
│   │       │    ....
|   |       ├── nxxxxx
│   │       │    ├── xxxxx.JPEG
│   │       │    ....
|   |       ....
```

## Preprocess the dataset
1. Clone the public DeepLearningExamples repository
```
cd $DEEP_LEARNING_EXAMPLE_DIR
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/MxNet/Classification/RN50v1.5
git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```

2. Build the ResNet50 MXNet NGC container
```
docker build . -t nvidia_rn50_mx
```

3. Start an interactive session in the NGC container to run the preprocessing
```
nvidia-docker run --rm -it --ipc=host \
-v $TRAIN_DIR:/data/train \
-v $VAL_DIR:/data/val \
-v $POSTPROCESS_DIR:/data/postprocess \
nvidia_rn50_mx
```

4. Run the preprocessing script in the container
```
./scripts/prepare_imagenet.sh /data /data/postprocess
``` 
Alternative to `prepare_imagenet.sh`, you can also run the python command directly in the container.
```
cd /data/postprocess
python /opt/mxnet/tools/im2rec.py --list --recursive train "/data/train"
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 train "/data/train"
python /opt/mxnet/tools/im2rec.py --list --recursive val "/data/val"
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 val "/data/val"
```

5. You should expect to see the following files in `/data/postprocess`:
```
./$DATADIR
│   ├── postprocess
|   |       ├── train.idx
|   |       ├── train.lst
|   |       ├── train.rec
|   |       ├── val.idx
|   |       ├── val.lst
|   |       └── val.rec
```

## Launch the training

1. Build the MLPerf Training container
```
cd $TRAINING_EXAMPLE_DIR
docker build --pull -t mlperf-nvidia:image_classification .
```

2. If you are using the same system
```
source config_7920T_2xA5000.sh
```
Else, you can create your own `config_SUT.sh` and `config_SUT_common.sh`
```
source config_SUT.sh
```

3. Run
```
CONT=mlperf-nvidia:image_classification DATADIR=$POSTPROCESS_DIR LOGDIR=$LOG_DIR ./run_with_docker.sh
```