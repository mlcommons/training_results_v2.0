cd ../pytorch
source config_NF5488A5.sh
CONT="<docker/registry>/mlperf-nvidia:rnn_speech_recognition-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> METADATA_DIR=<path/to/metadata/dir> SENTENCEPIECES_DIR=<path/to/sentencepieces/dir> bash ./run_with_docker.sh
